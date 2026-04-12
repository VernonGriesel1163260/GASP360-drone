from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

sys.path.append(str(Path(__file__).resolve().parent))
from common.logging_utils import setup_logger
from common.workspace import resolve_code_root, resolve_workspace_root


SCRIPT_NAME = "normalize_multistream_360"
PREPROCESS_METADATA_FILENAME = "_preprocess_metadata.json"
EXTRACTION_METADATA_FILENAME = "_extraction_metadata.json"
NORMALIZATION_METADATA_FILENAME = "_normalization_metadata.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
OUTPUT_FORMAT_ALIASES = {
    "dual-fisheye": "dfisheye",
    "dual_fisheye": "dfisheye",
    "dfisheye": "dfisheye",
    "fisheye_pair": "dfisheye",
    "flat": "flat",
}
LAYOUT_CHOICES = {"auto", "side_by_side_lr", "side_by_side_rl", "top_bottom_tb", "top_bottom_bt"}
ROTATE_CHOICES = {0, 90, 180, 270}


def code_root_from_script() -> Path:
    return resolve_code_root(__file__)


def workspace_root_from_script() -> Path:
    return resolve_workspace_root(caller_file=__file__)


def ensure_dirs(code_root: Path, workspace_root: Path, logger) -> dict[str, Path]:
    paths = {
        "code_root": code_root,
        "workspace_root": workspace_root,
        "data": workspace_root / "data",
        "frames_360": workspace_root / "data" / "frames_360",
        "frames_360_streams": workspace_root / "data" / "frames_360" / "streams",
        "logs": workspace_root / "logs",
        "preprocess_metadata": workspace_root / "data" / "input_video" / PREPROCESS_METADATA_FILENAME,
        "extraction_metadata": workspace_root / "data" / "frames_360" / EXTRACTION_METADATA_FILENAME,
        "normalization_metadata": workspace_root / "data" / "frames_360" / NORMALIZATION_METADATA_FILENAME,
    }

    for key in ("data", "frames_360", "frames_360_streams", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s -> %s", key, paths[key])

    return paths


def quote_cmd(cmd: Iterable[str]) -> str:
    return " ".join(f'"{c}"' if " " in c else c for c in cmd)


def load_json_file(path: Path, logger) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse JSON file %s: %s", path, exc)
        return None


def parse_stream_index_from_dirname(name: str) -> int | None:
    match = re.fullmatch(r"stream_(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def list_stream_dirs(streams_root: Path) -> list[dict]:
    results: list[dict] = []
    if not streams_root.exists():
        return results

    for child in sorted(streams_root.iterdir()):
        if not child.is_dir():
            continue
        stream_index = parse_stream_index_from_dirname(child.name)
        if stream_index is None:
            continue
        frame_files = sorted(p for p in child.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        results.append(
            {
                "stream_index": stream_index,
                "dir": child,
                "frame_count": len(frame_files),
                "frame_files": frame_files,
            }
        )
    return results


def normalize_output_format(value: str | None) -> str:
    raw = (value or "auto").strip().lower()
    if raw == "auto":
        return "auto"
    normalized = OUTPUT_FORMAT_ALIASES.get(raw, raw)
    if normalized not in {"dfisheye", "flat"}:
        raise ValueError(f"Unsupported normalized output format: {value}")
    return normalized


def resolve_output_prefix(value: str | None, default_prefix: str) -> str:
    candidate = (value or default_prefix).strip()
    if not candidate:
        raise ValueError("Output prefix cannot be empty.")
    return candidate


def remove_previous_outputs(frames_root: Path, output_prefix: str, metadata_path: Path, logger) -> int:
    removed = 0
    for image_path in frames_root.glob(f"{output_prefix}_*.jpg"):
        image_path.unlink()
        removed += 1
        logger.debug("Deleted old normalized frame: %s", image_path)
    if metadata_path.exists():
        metadata_path.unlink()
        removed += 1
        logger.debug("Deleted old normalization metadata: %s", metadata_path)
    return removed


def select_stream_pair(
    args,
    stream_dirs: list[dict],
    preprocess_payload: dict | None,
    extraction_payload: dict | None,
    logger,
) -> tuple[list[dict], str, dict | None]:
    if len(stream_dirs) < 2:
        raise RuntimeError("Normalizing multistream content requires at least two extracted stream folders under data/frames_360/streams.")

    stream_by_index = {item["stream_index"]: item for item in stream_dirs}
    recommendation = None

    if args.stream_pair:
        requested = [int(v) for v in args.stream_pair]
        missing = [idx for idx in requested if idx not in stream_by_index]
        if missing:
            available = sorted(stream_by_index)
            raise RuntimeError(
                f"Requested --stream-pair {requested} is not fully available. Missing: {missing}. Available streams: {available}"
            )
        return [stream_by_index[requested[0]], stream_by_index[requested[1]]], "explicit_stream_pair", recommendation

    if args.use_preprocess_recommendation and preprocess_payload:
        recommendation = preprocess_payload.get("recommendation") or {}
        pairwise_results = preprocess_payload.get("pairwise_results") or []
        compatible = [
            pair for pair in pairwise_results
            if pair.get("classification") in {"complementary_or_distinct", "same_scene_transformed"}
        ]
        if compatible:
            compatible.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
            best = compatible[0]
            stream_a = int(best["stream_a"])
            stream_b = int(best["stream_b"])
            if stream_a in stream_by_index and stream_b in stream_by_index:
                logger.info(
                    "Using preprocess-recommended pair %s vs %s (%s, confidence %.2f)",
                    stream_a,
                    stream_b,
                    best.get("classification"),
                    float(best.get("confidence") or 0.0),
                )
                return [stream_by_index[stream_a], stream_by_index[stream_b]], "preprocess_recommended_pair", recommendation

    if extraction_payload:
        selected_streams = extraction_payload.get("selected_streams") or []
        selected_indexes = [item.get("stream_index") for item in selected_streams if item.get("stream_index") is not None]
        candidate = [stream_by_index[idx] for idx in selected_indexes[:2] if idx in stream_by_index]
        if len(candidate) == 2:
            return candidate, "first_two_selected_streams", recommendation

    ordered = sorted(stream_dirs, key=lambda item: item["stream_index"])[:2]
    return ordered, "first_two_available_streams", recommendation



def infer_output_format(
    explicit_output_format: str,
    preprocess_payload: dict | None,
    selected_pair: list[dict],
    logger,
) -> tuple[str, str]:
    if explicit_output_format != "auto":
        return explicit_output_format, "explicit"

    stream_formats: dict[int, str | None] = {}
    if preprocess_payload:
        for stream in preprocess_payload.get("candidate_streams") or []:
            if stream.get("stream_index") is None:
                continue
            fmt = stream.get("effective_frame_format") or stream.get("frame_format_guess")
            stream_formats[int(stream["stream_index"])] = fmt

    pair_formats = [stream_formats.get(item["stream_index"]) for item in selected_pair]
    if all(fmt == "fisheye" for fmt in pair_formats):
        return "dfisheye", "auto_from_preprocess_pair_formats"

    # Fallback heuristic: if both source frames are square, dual-fisheye is a sensible normalized representation.
    sizes = []
    for item in selected_pair:
        frame_files = item["frame_files"]
        if not frame_files:
            continue
        with Image.open(frame_files[0]) as img:
            sizes.append(img.size)
    if len(sizes) == 2 and sizes[0][0] == sizes[0][1] and sizes[1][0] == sizes[1][1]:
        return "dfisheye", "auto_from_square_source_frames"

    logger.warning("Could not confidently infer normalized output format; falling back to flat.")
    return "flat", "auto_fallback_flat"



def resolve_layout(explicit_layout: str, resolved_output_format: str) -> tuple[str, str]:
    if explicit_layout != "auto":
        return explicit_layout, "explicit"
    if resolved_output_format == "dfisheye":
        return "side_by_side_lr", "auto_for_dfisheye"
    return "side_by_side_lr", "auto_default"



def apply_transforms(
    image: Image.Image,
    rotate_deg: int,
    flip_h: bool,
    flip_v: bool,
) -> Image.Image:
    result = image.copy()
    if rotate_deg not in ROTATE_CHOICES:
        raise ValueError(f"rotate_deg must be one of {sorted(ROTATE_CHOICES)}, got {rotate_deg}")
    if rotate_deg:
        # Pillow rotate is counter-clockwise; expand keeps full content.
        result = result.rotate(rotate_deg, expand=True)
    if flip_h:
        result = ImageOps.mirror(result)
    if flip_v:
        result = ImageOps.flip(result)
    return result



def harmonize_sizes(image_a: Image.Image, image_b: Image.Image, resize_mode: str) -> tuple[Image.Image, Image.Image, str | None]:
    if image_a.size == image_b.size:
        return image_a, image_b, None

    if resize_mode == "none":
        raise RuntimeError(
            f"Selected stream frames do not match in size: {image_a.size} vs {image_b.size}. Use --resize-streams-to match-first or max."
        )

    if resize_mode == "match-first":
        target_size = image_a.size
    elif resize_mode == "max":
        target_size = (max(image_a.width, image_b.width), max(image_a.height, image_b.height))
    else:
        raise ValueError(f"Unsupported resize mode: {resize_mode}")

    image_a_resized = image_a.resize(target_size, Image.Resampling.LANCZOS) if image_a.size != target_size else image_a
    image_b_resized = image_b.resize(target_size, Image.Resampling.LANCZOS) if image_b.size != target_size else image_b
    return image_a_resized, image_b_resized, f"resized_to_{target_size[0]}x{target_size[1]}_{resize_mode}"



def compose_pair(image_a: Image.Image, image_b: Image.Image, layout: str) -> Image.Image:
    if layout not in LAYOUT_CHOICES - {"auto"}:
        raise ValueError(f"Unsupported layout: {layout}")

    if layout == "side_by_side_lr":
        ordered = [image_a, image_b]
        canvas = Image.new("RGB", (ordered[0].width + ordered[1].width, max(ordered[0].height, ordered[1].height)))
        x = 0
        for img in ordered:
            y = (canvas.height - img.height) // 2
            canvas.paste(img, (x, y))
            x += img.width
        return canvas

    if layout == "side_by_side_rl":
        return compose_pair(image_b, image_a, "side_by_side_lr")

    if layout == "top_bottom_tb":
        ordered = [image_a, image_b]
        canvas = Image.new("RGB", (max(ordered[0].width, ordered[1].width), ordered[0].height + ordered[1].height))
        y = 0
        for img in ordered:
            x = (canvas.width - img.width) // 2
            canvas.paste(img, (x, y))
            y += img.height
        return canvas

    if layout == "top_bottom_bt":
        return compose_pair(image_b, image_a, "top_bottom_tb")

    raise ValueError(f"Unsupported layout: {layout}")



def list_matched_pairs(stream_a: dict, stream_b: dict, logger) -> list[tuple[Path, Path]]:
    files_a = {p.name: p for p in stream_a["frame_files"]}
    files_b = {p.name: p for p in stream_b["frame_files"]}
    common_names = sorted(set(files_a).intersection(files_b))
    if not common_names:
        raise RuntimeError(
            f"No matching frame filenames were found between stream {stream_a['stream_index']} and stream {stream_b['stream_index']}."
        )
    logger.info(
        "Matched %s paired frame(s) between stream %s and stream %s",
        len(common_names),
        stream_a["stream_index"],
        stream_b["stream_index"],
    )
    return [(files_a[name], files_b[name]) for name in common_names]



def derive_output_name(source_name: str, output_prefix: str, fallback_index: int) -> str:
    source_path = Path(source_name)
    match = re.search(r"_(\d+)$", source_path.stem)
    suffix = match.group(1) if match else f"{fallback_index:04d}"
    return f"{output_prefix}_{suffix}.jpg"



def count_flat_frames(frames_root: Path, output_prefix: str) -> int:
    return sum(1 for p in frames_root.glob(f"{output_prefix}_*.jpg") if p.is_file())



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize paired multistream extractions into a flat frame set under data/frames_360 so downstream reprojection can consume them."
    )
    parser.add_argument("--mode", choices=["auto", "explicit"], default="auto", help="How to resolve pair/layout/format. Manual overrides always win.")
    parser.add_argument("--stream-pair", nargs=2, type=int, default=None, help="Explicit stream indexes to combine, for example --stream-pair 0 1")
    parser.add_argument("--use-preprocess-recommendation", action="store_true", help="Use preprocess pairwise recommendations to pick the best stream pair when possible.")
    parser.add_argument("--output-format", type=str, default="auto", help="Normalized output frame format: auto, dfisheye, dual-fisheye, or flat.")
    parser.add_argument("--layout", choices=sorted(LAYOUT_CHOICES), default="auto", help="How to place the paired frames on the normalized canvas.")
    parser.add_argument("--resize-streams-to", choices=["match-first", "max", "none"], default="match-first", help="How to handle mismatched input frame sizes across the selected pair.")
    parser.add_argument("--rotate-a", type=int, default=0, choices=sorted(ROTATE_CHOICES), help="Rotate stream A frames by 0/90/180/270 degrees before composing.")
    parser.add_argument("--rotate-b", type=int, default=0, choices=sorted(ROTATE_CHOICES), help="Rotate stream B frames by 0/90/180/270 degrees before composing.")
    parser.add_argument("--flip-h-a", action="store_true")
    parser.add_argument("--flip-v-a", action="store_true")
    parser.add_argument("--flip-h-b", action="store_true")
    parser.add_argument("--flip-v-b", action="store_true")
    parser.add_argument("--output-prefix", type=str, default=None, help="Output frame prefix written into data/frames_360. Defaults to the extraction frame prefix.")
    parser.add_argument("--limit", type=int, default=None, help="Limit how many paired frames are normalized.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    code_root = code_root_from_script()
    workspace_root = workspace_root_from_script()
    logger, run_log_path, latest_log_path = setup_logger(
        SCRIPT_NAME,
        verbose=args.verbose,
        workspace_root=workspace_root,
    )

    try:
        logger.info("Code root: %s", code_root)
        logger.info("Workspace root: %s", workspace_root)
        logger.info("Run log: %s", run_log_path)
        logger.info("Latest log: %s", latest_log_path)

        paths = ensure_dirs(code_root, workspace_root, logger)
        logger.info("Project folders checked and created if missing")

        preprocess_payload = load_json_file(paths["preprocess_metadata"], logger)
        extraction_payload = load_json_file(paths["extraction_metadata"], logger)

        stream_dirs = list_stream_dirs(paths["frames_360_streams"])
        logger.info("Available extracted multistream folders: %s", [item["stream_index"] for item in stream_dirs])

        selected_pair, pair_selection_mode, preprocess_recommendation = select_stream_pair(
            args=args,
            stream_dirs=stream_dirs,
            preprocess_payload=preprocess_payload,
            extraction_payload=extraction_payload,
            logger=logger,
        )

        logger.info("Selected pair mode: %s", pair_selection_mode)
        logger.info("Selected pair stream indexes: %s", [item["stream_index"] for item in selected_pair])

        resolved_output_format, output_format_source = infer_output_format(
            explicit_output_format=normalize_output_format(args.output_format),
            preprocess_payload=preprocess_payload,
            selected_pair=selected_pair,
            logger=logger,
        )
        resolved_layout, layout_source = resolve_layout(args.layout, resolved_output_format)

        output_prefix_default = extraction_payload.get("frame_prefix") if extraction_payload else "frame360"
        output_prefix = resolve_output_prefix(args.output_prefix, output_prefix_default)

        logger.info("Resolved normalized output format: %s (%s)", resolved_output_format, output_format_source)
        logger.info("Resolved normalized layout: %s (%s)", resolved_layout, layout_source)
        logger.info("Output prefix: %s", output_prefix)

        if args.clean:
            removed = remove_previous_outputs(paths["frames_360"], output_prefix, paths["normalization_metadata"], logger)
            logger.info("Removed %s previous normalization artifact(s)", removed)

        matched_pairs = list_matched_pairs(selected_pair[0], selected_pair[1], logger)
        if args.limit is not None:
            matched_pairs = matched_pairs[: args.limit]
            logger.info("Applying limit: normalizing first %s paired frame(s)", len(matched_pairs))

        output_records: list[dict] = []
        size_actions: list[str] = []

        for idx, (path_a, path_b) in enumerate(matched_pairs, start=1):
            output_name = derive_output_name(path_a.name, output_prefix, idx)
            output_path = paths["frames_360"] / output_name

            if output_path.exists() and not args.overwrite:
                logger.info("Skipping existing normalized frame (use --overwrite to replace): %s", output_path)
                output_records.append(
                    {
                        "index": idx,
                        "source_a": str(path_a),
                        "source_b": str(path_b),
                        "output": str(output_path),
                        "skipped_existing": True,
                    }
                )
                continue

            with Image.open(path_a) as img_a_src, Image.open(path_b) as img_b_src:
                img_a = apply_transforms(img_a_src.convert("RGB"), args.rotate_a, args.flip_h_a, args.flip_v_a)
                img_b = apply_transforms(img_b_src.convert("RGB"), args.rotate_b, args.flip_h_b, args.flip_v_b)
                img_a, img_b, size_action = harmonize_sizes(img_a, img_b, args.resize_streams_to)
                if size_action:
                    size_actions.append(size_action)
                composed = compose_pair(img_a, img_b, resolved_layout)
                composed.save(output_path, quality=95)

            output_records.append(
                {
                    "index": idx,
                    "source_a": str(path_a),
                    "source_b": str(path_b),
                    "output": str(output_path),
                    "output_name": output_name,
                    "skipped_existing": False,
                }
            )
            logger.info("Wrote normalized frame %s -> %s", idx, output_path)

        output_frame_count = count_flat_frames(paths["frames_360"], output_prefix)
        effective_convert_input_format = "dual-fisheye" if resolved_output_format == "dfisheye" else resolved_output_format

        normalization_payload = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_script": SCRIPT_NAME,
            "mode": args.mode,
            "pair_selection_mode": pair_selection_mode,
            "preprocess_metadata_path": str(paths["preprocess_metadata"]),
            "preprocess_recommendation": preprocess_recommendation,
            "extraction_metadata_path": str(paths["extraction_metadata"]),
            "selected_pair": [
                {
                    "stream_index": item["stream_index"],
                    "dir": str(item["dir"]),
                    "frame_count": item["frame_count"],
                }
                for item in selected_pair
            ],
            "resolved_output_format": resolved_output_format,
            "output_format_source": output_format_source,
            "resolved_layout": resolved_layout,
            "layout_source": layout_source,
            "effective_convert_input_format": effective_convert_input_format,
            "output_prefix": output_prefix,
            "output_dir": str(paths["frames_360"]),
            "output_frame_count": output_frame_count,
            "resize_streams_to": args.resize_streams_to,
            "size_actions_observed": sorted(set(size_actions)),
            "transforms": {
                "stream_a": {"rotate": args.rotate_a, "flip_h": args.flip_h_a, "flip_v": args.flip_v_a},
                "stream_b": {"rotate": args.rotate_b, "flip_h": args.flip_h_b, "flip_v": args.flip_v_b},
            },
            "outputs_preview": output_records[:10],
            "notes": [
                "Normalized multistream frames are written into data/frames_360 as a flat frame set so convert_360_to_views.py can consume them.",
                "When resolved_output_format is dfisheye, run convert_360_to_views.py with --input-format auto or dual-fisheye/dfisheye so FFmpeg v360 interprets the normalized frames correctly.",
            ],
        }

        paths["normalization_metadata"].write_text(json.dumps(normalization_payload, indent=2), encoding="utf-8")
        logger.info("Wrote normalization metadata: %s", paths["normalization_metadata"])
        logger.info("Normalized frame count: %s", output_frame_count)
        logger.info("Effective convert input format: %s", effective_convert_input_format)

        print("\n=== Normalize Multistream Report ===")
        print(f"Mode: {args.mode}")
        print(f"Selected pair: {[item['stream_index'] for item in selected_pair]}")
        print(f"Resolved output format: {resolved_output_format}")
        print(f"Resolved layout: {resolved_layout}")
        print(f"Effective convert input format: {effective_convert_input_format}")
        print(f"Output prefix: {output_prefix}")
        print(f"Normalized frame count: {output_frame_count}")
        print(f"Metadata: {paths['normalization_metadata']}")

        return 0

    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
