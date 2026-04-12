from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))

from common.logging_utils import setup_logger
from common.workspace import resolve_code_root, resolve_workspace_root
from common.presets import DEFAULT_PRESET_NAME, get_preset, get_preset_names
from common.config_merge import deep_merge_dict, build_projection_override_dict


SCRIPT_NAME = "convert_360_to_views"
PROJECTION_FORMAT_ALIASES = {
    "e": "equirect",
    "equirect": "equirect",
    "equirectangular": "equirect",
    "flat": "flat",
    "plain": "flat",
    "gnomonic": "flat",
    "rectilinear": "flat",
    "fisheye": "fisheye",
    "fish-eye": "fisheye",
    "fish_eye": "fisheye",
    "fishere": "fisheye",
    "fishereye": "fisheye",
    "dfisheye": "dfisheye",
    "dual-fisheye": "dfisheye",
    "dual_fisheye": "dfisheye",
    "dualfisheye": "dfisheye",
    "eac": "eac",
    "equiangular-cubemap": "eac",
    "equiangular_cubemap": "eac",
    "c3x2": "c3x2",
    "cubemap": "c3x2",
    "cubemap-3x2": "c3x2",
    "cubemap_3x2": "c3x2",
    "c6x1": "c6x1",
    "cubemap-6x1": "c6x1",
    "cubemap_6x1": "c6x1",
    "c1x6": "c1x6",
    "cubemap-1x6": "c1x6",
    "cubemap_1x6": "c1x6",
    "barrel": "barrel",
    "fb": "fb",
    "barrelsplit": "barrelsplit",
    "sg": "sg",
    "stereographic": "sg",
    "little-planet": "sg",
    "little_planet": "sg",
    "mercator": "mercator",
    "ball": "ball",
    "hammer": "hammer",
    "sinusoidal": "sinusoidal",
    "pannini": "pannini",
    "cylindrical": "cylindrical",
    "perspective": "perspective",
    "tetrahedron": "tetrahedron",
    "tsp": "tsp",
    "he": "he",
    "hequirect": "he",
    "half-equirect": "he",
    "half_equirect": "he",
    "half-equirectangular": "he",
    "half_equirectangular": "he",
    "equisolid": "equisolid",
    "og": "og",
    "orthographic": "og",
    "octahedron": "octahedron",
    "cylindricalea": "cylindricalea",
}
OUTPUT_ONLY_FORMATS = {"perspective"}
AUTO_FORMAT_SENTINEL = "auto"
PROJECTION_METADATA_FILENAME = "_projection_metadata.json"
INPUT_FOV_RECOMMENDED_FORMATS = {
    "flat",
    "fisheye",
    "dfisheye",
    "sg",
    "equisolid",
    "og",
    "cylindrical",
    "cylindricalea",
    "pannini",
}


def code_root_from_script() -> Path:
    return resolve_code_root(__file__)


def workspace_root_from_script() -> Path:
    return resolve_workspace_root(caller_file=__file__)


def normalize_yaw(yaw: float) -> float:
    while yaw > 180:
        yaw -= 360
    while yaw < -180:
        yaw += 360
    return yaw


def normalize_projection_format(value: str | None, *, default: str = "equirect") -> str:
    if value is None:
        return default

    key = value.strip().lower().replace(" ", "-")
    canonical = PROJECTION_FORMAT_ALIASES.get(key)
    if canonical is None:
        supported = ", ".join(sorted(PROJECTION_FORMAT_ALIASES.keys()))
        raise ValueError(f"Unsupported projection format '{value}'. Supported names/aliases: {supported}")

    if canonical in OUTPUT_ONLY_FORMATS:
        raise ValueError(
            f"Projection format '{value}' resolves to '{canonical}', which is output-only in FFmpeg v360 and cannot be used as an input format."
        )

    return canonical


def region_mean(arr: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> float:
    h, w = arr.shape
    xs0 = max(0, min(w - 1, int(round(x0 * w))))
    xs1 = max(xs0 + 1, min(w, int(round(x1 * w))))
    ys0 = max(0, min(h - 1, int(round(y0 * h))))
    ys1 = max(ys0 + 1, min(h, int(round(y1 * h))))
    return float(arr[ys0:ys1, xs0:xs1].mean())


def load_preview_luma(frame_path: Path, size: int = 96) -> tuple[np.ndarray, int, int]:
    with Image.open(frame_path) as image:
        width, height = image.size
        preview = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(preview, dtype=np.float32) / 255.0
    return arr, width, height


def detect_input_projection_format(frame_path: Path, logger) -> tuple[str, float, list[str], dict[str, float | int | str]]:
    arr, width, height = load_preview_luma(frame_path)
    ratio = width / max(height, 1)

    corner_mean = float(np.mean([
        region_mean(arr, 0.00, 0.16, 0.00, 0.16),
        region_mean(arr, 0.84, 1.00, 0.00, 0.16),
        region_mean(arr, 0.00, 0.16, 0.84, 1.00),
        region_mean(arr, 0.84, 1.00, 0.84, 1.00),
    ]))
    center_mean = region_mean(arr, 0.40, 0.60, 0.40, 0.60)
    top_mid_mean = region_mean(arr, 0.40, 0.60, 0.00, 0.14)
    bottom_mid_mean = region_mean(arr, 0.40, 0.60, 0.86, 1.00)
    left_lens_mean = region_mean(arr, 0.12, 0.38, 0.24, 0.76)
    right_lens_mean = region_mean(arr, 0.62, 0.88, 0.24, 0.76)

    metrics = {
        "frame_width": width,
        "frame_height": height,
        "aspect_ratio": ratio,
        "corner_mean": corner_mean,
        "center_mean": center_mean,
        "top_mid_mean": top_mid_mean,
        "bottom_mid_mean": bottom_mid_mean,
        "left_lens_mean": left_lens_mean,
        "right_lens_mean": right_lens_mean,
    }

    reasons = [
        f"frame={frame_path.name}",
        f"size={width}x{height}",
        f"aspect_ratio={ratio:.3f}",
        f"corner_mean={corner_mean:.3f}",
        f"center_mean={center_mean:.3f}",
        f"top_mid_mean={top_mid_mean:.3f}",
        f"bottom_mid_mean={bottom_mid_mean:.3f}",
        f"left_lens_mean={left_lens_mean:.3f}",
        f"right_lens_mean={right_lens_mean:.3f}",
    ]

    if 1.45 <= ratio <= 1.55:
        reasons.append("Aspect ratio is close to 3:2, which often matches a 3x2 cubemap layout.")
        return "c3x2", 0.92, reasons, metrics

    if 5.5 <= ratio <= 6.5:
        reasons.append("Aspect ratio is close to 6:1, which often matches a 6x1 cubemap layout.")
        return "c6x1", 0.92, reasons, metrics

    if 0.14 <= ratio <= 0.19:
        reasons.append("Aspect ratio is close to 1:6, which often matches a 1x6 cubemap layout.")
        return "c1x6", 0.92, reasons, metrics

    looks_like_dual_fisheye = (
        1.70 <= ratio <= 2.30
        and corner_mean < 0.12
        and top_mid_mean < 0.18
        and bottom_mid_mean < 0.18
        and left_lens_mean > 0.20
        and right_lens_mean > 0.20
    )
    if looks_like_dual_fisheye:
        reasons.append("Two bright lens regions with dark corners/top/bottom suggest a side-by-side dual-fisheye frame.")
        return "dfisheye", 0.86, reasons, metrics

    looks_like_fisheye = (
        0.80 <= ratio <= 1.25
        and corner_mean < 0.12
        and center_mean > corner_mean + 0.12
    )
    if looks_like_fisheye:
        reasons.append("Near-square frame with dark corners and a bright center suggests a single fisheye frame.")
        return "fisheye", 0.82, reasons, metrics

    looks_like_equirect = (
        1.70 <= ratio <= 2.30
        and corner_mean > 0.10
    )
    if looks_like_equirect:
        reasons.append("2:1 frame with image content present in the corners suggests an equirectangular panorama.")
        return "equirect", 0.78, reasons, metrics

    reasons.append("Falling back to flat because the frame does not strongly match the common 360 layouts.")
    return "flat", 0.55, reasons, metrics


def resolve_input_projection_format(requested_format: str | None, sample_frame: Path, logger) -> dict[str, object]:
    raw_value = (requested_format or "equirect").strip()

    if raw_value.lower() != AUTO_FORMAT_SENTINEL:
        return {
            "requested_input_format": raw_value,
            "requested_input_format_normalized": raw_value.lower(),
            "resolved_input_format": normalize_projection_format(raw_value),
            "detection_mode": "manual",
            "auto_detection": None,
        }

    detected_format, confidence, reasons, metrics = detect_input_projection_format(sample_frame, logger)

    logger.info(
        "Auto-detected input format: %s (confidence %.2f) using sample frame %s",
        detected_format,
        confidence,
        sample_frame.name,
    )
    for reason in reasons:
        logger.info("Auto-detect detail: %s", reason)

    if detected_format in {"c3x2", "c6x1", "c1x6"}:
        logger.warning(
            "Auto-detect inferred a cubemap layout. If this looks wrong, override with --input-format manually because auto-detect is heuristic-based."
        )

    return {
        "requested_input_format": raw_value,
        "requested_input_format_normalized": raw_value.lower(),
        "resolved_input_format": detected_format,
        "detection_mode": "auto",
        "auto_detection": {
            "sample_frame_name": sample_frame.name,
            "sample_frame_path": str(sample_frame),
            "confidence": confidence,
            "reasons": reasons,
            "metrics": metrics,
        },
    }


def build_projection_metadata(
    *,
    requested_format: str | None,
    resolution_info: dict[str, object],
    input_frames: list[Path],
    paths: dict[str, Path],
    ffmpeg_path: Path,
    preset_name: str,
    input_prefix: str | None,
    input_h_fov: float | None,
    input_v_fov: float | None,
    input_d_fov: float | None,
    views: list[str],
    view_yaws: dict[str, float],
    projection_config: dict[str, object],
    output_d_fov: float | None,
) -> dict[str, object]:
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_script": SCRIPT_NAME,
        "frames_360_dir": str(paths["frames_360"]),
        "projection_metadata_path": str(paths["frames_360"] / PROJECTION_METADATA_FILENAME),
        "preset": preset_name,
        "input_prefix": input_prefix,
        "input_frame_count": len(input_frames),
        "input_frame_names_preview": [p.name for p in input_frames[:10]],
        "requested_input_format": requested_format or "equirect",
        "resolved_input_format": resolution_info["resolved_input_format"],
        "detection_mode": resolution_info["detection_mode"],
        "auto_detection": resolution_info["auto_detection"],
        "input_fov": {
            "input_h_fov": input_h_fov,
            "input_v_fov": input_v_fov,
            "input_d_fov": input_d_fov,
        },
        "output_projection": {
            "format": "flat",
            "views": views,
            "view_yaws": {k: float(v) for k, v in view_yaws.items() if k in views},
            "h_fov": projection_config["h_fov"],
            "v_fov": projection_config["v_fov"],
            "d_fov": output_d_fov,
            "width": projection_config["width"],
            "height": projection_config["height"],
            "pitch": projection_config["pitch"],
            "roll": projection_config["roll"],
            "interpolation": projection_config["interpolation"],
            "quality": projection_config["quality"],
        },
        "tools": {
            "ffmpeg_path": str(ffmpeg_path),
        },
    }
    return metadata


def write_projection_metadata(metadata_path: Path, payload: dict[str, object], logger) -> None:
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote projection metadata sidecar: %s", metadata_path)


def ensure_dirs(code_root: Path, workspace_root: Path, logger) -> dict[str, Path]:
    paths = {
        "code_root": code_root,
        "workspace_root": workspace_root,
        "data": workspace_root / "data",
        "frames_360": workspace_root / "data" / "frames_360",
        "frames_perspective": workspace_root / "data" / "frames_perspective",
        "logs": workspace_root / "logs",
        "tools": code_root / "tools",
        "scripts": code_root / "scripts",
    }

    for key in ("data", "frames_360", "frames_perspective", "logs"):
        paths[key].mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s -> %s", key, paths[key])

    return paths


def ensure_view_dirs(paths: dict[str, Path], views: list[str], logger) -> dict[str, Path]:
    view_dirs = {}
    for view_name in views:
        view_dir = paths["frames_perspective"] / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        view_dirs[view_name] = view_dir
        logger.debug("Ensured view directory exists: %s -> %s", view_name, view_dir)
    return view_dirs


def find_ffmpeg(code_root: Path) -> Path:
    candidates = [
        code_root / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
        code_root / "tools" / "ffmpeg.exe",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    ffmpeg_on_path = shutil.which("ffmpeg")
    if ffmpeg_on_path:
        return Path(ffmpeg_on_path)

    raise FileNotFoundError(
        "FFmpeg was not found. Expected one of:\n"
        f"  - {code_root / 'tools' / 'ffmpeg' / 'bin' / 'ffmpeg.exe'}\n"
        f"  - {code_root / 'tools' / 'ffmpeg.exe'}\n"
        "  - ffmpeg on PATH"
    )


def quote_cmd(cmd: Iterable[str]) -> str:
    return " ".join(f'"{c}"' if " " in c else c for c in cmd)


def stream_process_output(process: subprocess.Popen, logger, prefix: str, verbose: bool) -> None:
    assert process.stdout is not None

    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if not line:
            continue

        logger.debug("%s %s", prefix, line)
        if verbose:
            print(f"{prefix} {line}")


def run_command_streaming(cmd: list[str], logger, verbose: bool = False) -> None:
    logger.info("Running command")
    logger.debug("Command: %s", quote_cmd(cmd))

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        shell=False,
    )

    try:
        stream_process_output(process, logger, prefix="[SUBPROCESS]", verbose=verbose)
    finally:
        return_code = process.wait()

    logger.debug("Command exit code: %s", return_code)

    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}")


def find_input_frames(frames_360_dir: Path, prefix: str | None, logger) -> list[Path]:
    candidates = sorted(frames_360_dir.glob("*.jpg")) + sorted(frames_360_dir.glob("*.jpeg")) + sorted(frames_360_dir.glob("*.png"))
    candidates = sorted(set(candidates))

    if prefix:
        candidates = [p for p in candidates if p.stem.startswith(prefix)]
        logger.debug("Filtered input frames by prefix '%s': %s file(s)", prefix, len(candidates))

    return candidates


def output_name_for_view(input_frame: Path, view_name: str) -> str:
    return f"{input_frame.stem}_{view_name}.jpg"


def clear_existing_outputs(output_dir: Path, logger) -> int:
    removed = 0
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        for file_path in output_dir.glob(pattern):
            file_path.unlink()
            removed += 1
            logger.debug("Deleted old output file: %s", file_path)
    return removed


def build_v360_command(
    ffmpeg_path: Path,
    input_frame: Path,
    output_frame: Path,
    input_format: str,
    yaw: float,
    h_fov: float,
    v_fov: float,
    d_fov: float | None,
    out_width: int,
    out_height: int,
    pitch: float,
    roll: float,
    interpolation: str,
    overwrite: bool,
    jpg_quality: int,
    input_h_fov: float | None,
    input_v_fov: float | None,
    input_d_fov: float | None,
) -> list[str]:
    vf_parts = [
        f"input={input_format}",
        "output=flat",
        f"yaw={yaw}",
        f"pitch={pitch}",
        f"roll={roll}",
    ]

    if d_fov is not None:
        vf_parts.append(f"d_fov={d_fov}")
    else:
        vf_parts.append(f"h_fov={h_fov}")
        vf_parts.append(f"v_fov={v_fov}")

    if input_d_fov is not None:
        vf_parts.append(f"id_fov={input_d_fov}")
    else:
        if input_h_fov is not None:
            vf_parts.append(f"ih_fov={input_h_fov}")
        if input_v_fov is not None:
            vf_parts.append(f"iv_fov={input_v_fov}")

    vf_parts.extend(
        [
            f"w={out_width}",
            f"h={out_height}",
            f"interp={interpolation}",
        ]
    )
    vf = "v360=" + ":".join(vf_parts)

    cmd = [str(ffmpeg_path)]
    cmd.append("-y" if overwrite else "-n")
    cmd.extend(
        [
            "-i",
            str(input_frame),
            "-vf",
            vf,
            "-frames:v",
            "1",
            "-update",
            "1",
            "-q:v",
            str(jpg_quality),
            str(output_frame),
        ]
    )
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 360 frames from multiple source projections into perspective view images."
    )

    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET_NAME,
        choices=get_preset_names(),
        help="Projection preset name.",
    )
    parser.add_argument(
        "--input-prefix",
        type=str,
        default=None,
        help="Only process input frames whose stem starts with this prefix.",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        default=None,
        help=(
            "Input projection format for FFmpeg v360. Examples: auto, equirect, fisheye, dual-fisheye, flat/plain, eac, cubemap. "
            "Defaults to equirect if omitted."
        ),
    )
    parser.add_argument("--input-h-fov", type=float, default=None, help="Optional input horizontal FOV in degrees.")
    parser.add_argument("--input-v-fov", type=float, default=None, help="Optional input vertical FOV in degrees.")
    parser.add_argument("--input-d-fov", type=float, default=None, help="Optional input diagonal FOV in degrees.")
    parser.add_argument(
        "--views",
        nargs="+",
        default=None,
        help="Override views from preset.",
    )
    parser.add_argument("--h-fov", type=float, default=None, help="Output horizontal FOV in degrees.")
    parser.add_argument("--v-fov", type=float, default=None, help="Output vertical FOV in degrees.")
    parser.add_argument("--d-fov", type=float, default=None, help="Output diagonal FOV in degrees. Overrides h/v if set.")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--pitch", type=float, default=None)
    parser.add_argument("--roll", type=float, default=None)
    parser.add_argument("--interpolation", type=str, default=None)
    parser.add_argument("--quality", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
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

        preset = get_preset(args.preset)
        projection_config = deep_merge_dict(
            preset["projection"],
            build_projection_override_dict(args),
        )

        requested_input_format = args.input_format or preset["projection"].get("input_format", "equirect")
        input_h_fov = args.input_h_fov if args.input_h_fov is not None else preset["projection"].get("input_h_fov")
        input_v_fov = args.input_v_fov if args.input_v_fov is not None else preset["projection"].get("input_v_fov")
        input_d_fov = args.input_d_fov if args.input_d_fov is not None else preset["projection"].get("input_d_fov")
        output_d_fov = args.d_fov if args.d_fov is not None else preset["projection"].get("d_fov")

        views = projection_config["views"]
        view_yaws = projection_config["view_yaws"]

        missing_view_yaws = [v for v in views if v not in view_yaws]
        if missing_view_yaws:
            logger.error("Preset/config missing yaw entries for views: %s", missing_view_yaws)
            return 1

        view_dirs = ensure_view_dirs(paths, views, logger)

        ffmpeg_path = find_ffmpeg(code_root)
        logger.info("FFmpeg found: %s", ffmpeg_path)

        input_frames = find_input_frames(paths["frames_360"], args.input_prefix, logger)
        if not input_frames:
            logger.error("No input frames found in %s", paths["frames_360"])
            return 1

        if args.limit is not None:
            input_frames = input_frames[: args.limit]
            logger.info("Applying limit: processing first %s frame(s)", len(input_frames))

        input_format_resolution = resolve_input_projection_format(
            requested_format=requested_input_format,
            sample_frame=input_frames[0],
            logger=logger,
        )
        input_format = input_format_resolution["resolved_input_format"]

        if input_format in INPUT_FOV_RECOMMENDED_FORMATS and all(
            v is None for v in (input_h_fov, input_v_fov, input_d_fov)
        ):
            logger.warning(
                "Input format '%s' often needs input FOV values for accurate reprojection. Consider --input-h-fov/--input-v-fov or --input-d-fov.",
                input_format,
            )

        if input_format == "flat" and len(views) > 1:
            logger.warning(
                "Input format is 'flat' and multiple output views were requested. For ordinary non-360 video, using a single view such as '--views front' is usually safer."
            )

        projection_metadata = build_projection_metadata(
            requested_format=requested_input_format,
            resolution_info=input_format_resolution,
            input_frames=input_frames,
            paths=paths,
            ffmpeg_path=ffmpeg_path,
            preset_name=args.preset,
            input_prefix=args.input_prefix,
            input_h_fov=input_h_fov,
            input_v_fov=input_v_fov,
            input_d_fov=input_d_fov,
            views=views,
            view_yaws=view_yaws,
            projection_config=projection_config,
            output_d_fov=output_d_fov,
        )
        write_projection_metadata(paths["frames_360"] / PROJECTION_METADATA_FILENAME, projection_metadata, logger)

        logger.info("Preset: %s", args.preset)
        logger.info("Found %s input frame(s)", len(input_frames))
        logger.info("Views to generate: %s", views)
        logger.info(
            "Input projection: requested_format=%s resolved_format=%s detection_mode=%s input_h_fov=%s input_v_fov=%s input_d_fov=%s",
            requested_input_format,
            input_format,
            input_format_resolution["detection_mode"],
            input_h_fov,
            input_v_fov,
            input_d_fov,
        )
        logger.info(
            "Output projection: format=flat h_fov=%s v_fov=%s d_fov=%s width=%s height=%s pitch=%.2f roll=%.2f interp=%s quality=%s",
            projection_config["h_fov"],
            projection_config["v_fov"],
            output_d_fov,
            projection_config["width"],
            projection_config["height"],
            projection_config["pitch"],
            projection_config["roll"],
            projection_config["interpolation"],
            projection_config["quality"],
        )

        if args.clean:
            total_removed = 0
            for view_name in views:
                removed = clear_existing_outputs(view_dirs[view_name], logger)
                total_removed += removed
                logger.info("Removed %s existing file(s) from %s", removed, view_dirs[view_name])
            logger.info("Total removed output files: %s", total_removed)

        total_outputs = 0

        for index, input_frame in enumerate(input_frames, start=1):
            logger.info("Processing frame %s / %s: %s", index, len(input_frames), input_frame.name)

            for view_name in views:
                yaw = normalize_yaw(view_yaws[view_name])
                output_frame = view_dirs[view_name] / output_name_for_view(input_frame, view_name)

                logger.debug(
                    "Generating view '%s' from %s -> %s",
                    view_name,
                    input_frame,
                    output_frame,
                )

                cmd = build_v360_command(
                    ffmpeg_path=ffmpeg_path,
                    input_frame=input_frame,
                    output_frame=output_frame,
                    input_format=input_format,
                    yaw=yaw,
                    h_fov=projection_config["h_fov"],
                    v_fov=projection_config["v_fov"],
                    d_fov=output_d_fov,
                    out_width=projection_config["width"],
                    out_height=projection_config["height"],
                    pitch=projection_config["pitch"],
                    roll=projection_config["roll"],
                    interpolation=projection_config["interpolation"],
                    overwrite=args.overwrite,
                    jpg_quality=projection_config["quality"],
                    input_h_fov=input_h_fov,
                    input_v_fov=input_v_fov,
                    input_d_fov=input_d_fov,
                )

                run_command_streaming(cmd, logger, verbose=args.verbose)
                total_outputs += 1

        logger.info("Conversion completed successfully")
        logger.info("Generated %s output image(s)", total_outputs)

        for view_name in views:
            count = len(list(view_dirs[view_name].glob("*.jpg")))
            logger.info("Output count in %s: %s", view_name, count)

        return 0

    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
