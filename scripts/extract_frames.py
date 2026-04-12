from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

sys.path.append(str(Path(__file__).resolve().parent))
from common.logging_utils import setup_logger
from common.workspace import resolve_code_root, resolve_workspace_root


SCRIPT_NAME = "extract_frames"
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
    ".webm",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".ts",
    ".mts",
    ".m2ts",
    ".insv",
    ".osv",
    ".360",
}
EXTRACTION_METADATA_FILENAME = "_extraction_metadata.json"
PREPROCESS_METADATA_FILENAME = "_preprocess_metadata.json"


def code_root_from_script() -> Path:
    return resolve_code_root(__file__)


def workspace_root_from_script() -> Path:
    return resolve_workspace_root(caller_file=__file__)


def ensure_dirs(code_root: Path, workspace_root: Path, logger) -> dict[str, Path]:
    paths = {
        "code_root": code_root,
        "workspace_root": workspace_root,
        "data": workspace_root / "data",
        "input_video": workspace_root / "data" / "input_video",
        "frames_360": workspace_root / "data" / "frames_360",
        "frames_360_streams": workspace_root / "data" / "frames_360" / "streams",
        "frames_perspective": workspace_root / "data" / "frames_perspective",
        "colmap": workspace_root / "data" / "colmap",
        "logs": workspace_root / "logs",
        "tools": code_root / "tools",
        "scripts": code_root / "scripts",
        "extraction_metadata": workspace_root / "data" / "frames_360" / EXTRACTION_METADATA_FILENAME,
        "preprocess_metadata": workspace_root / "data" / "input_video" / PREPROCESS_METADATA_FILENAME,
    }

    for key in (
        "data",
        "input_video",
        "frames_360",
        "frames_360_streams",
        "frames_perspective",
        "colmap",
        "logs",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s -> %s", key, paths[key])

    for view_name in ("front", "right", "back", "left"):
        view_dir = paths["frames_perspective"] / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured perspective subdirectory exists: %s", view_dir)

    return paths


def find_ffmpeg(code_root: Path) -> Path:
    candidates = [
        code_root / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
        code_root / "tools" / "ffmpeg" / "bin" / "ffmpeg",
        code_root / "tools" / "ffmpeg.exe",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    ffmpeg_on_path = shutil.which("ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg") or shutil.which("ffmpeg")
    if ffmpeg_on_path:
        return Path(ffmpeg_on_path).resolve()

    raise FileNotFoundError(
        "FFmpeg was not found. Expected one of:\n"
        f"  - {code_root / 'tools' / 'ffmpeg' / 'bin' / 'ffmpeg.exe'}\n"
        f"  - {code_root / 'tools' / 'ffmpeg.exe'}\n"
        "  - ffmpeg on PATH"
    )


def find_ffprobe(ffmpeg_path: Path) -> Path | None:
    local_names = ["ffprobe.exe", "ffprobe"]
    for local_name in local_names:
        local_ffprobe = ffmpeg_path.parent / local_name
        if local_ffprobe.exists():
            return local_ffprobe.resolve()

    ffprobe_on_path = shutil.which("ffprobe.exe" if sys.platform.startswith("win") else "ffprobe") or shutil.which("ffprobe")
    if ffprobe_on_path:
        return Path(ffprobe_on_path).resolve()

    return None


def quote_cmd(cmd: Iterable[str]) -> str:
    return " ".join(f'"{c}"' if " " in c else c for c in cmd)


def stream_process_output(
    process: subprocess.Popen,
    logger,
    prefix: str,
    verbose: bool,
) -> None:
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
    )

    try:
        stream_process_output(process, logger, "[SUBPROCESS]", verbose=verbose)
    finally:
        return_code = process.wait()

    logger.debug("Command exit code: %s", return_code)

    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}")


def run_command_capture(cmd: list[str], logger) -> subprocess.CompletedProcess:
    logger.debug("Running command: %s", quote_cmd(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.stdout:
        logger.debug("[COMMAND STDOUT] %s", result.stdout.strip())
    if result.stderr:
        logger.debug("[COMMAND STDERR] %s", result.stderr.strip())
    return result


def ffprobe_duration(ffprobe_path: Path, input_video: Path, logger) -> float | None:
    cmd = [
        str(ffprobe_path),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_video),
    ]

    logger.debug("Running ffprobe for duration: %s", quote_cmd(cmd))

    try:
        result = run_command_capture(cmd, logger)
        if result.returncode != 0:
            logger.error("ffprobe failed with exit code %s", result.returncode)
            return None

        duration = float(result.stdout.strip())
        logger.debug("Detected duration: %.4f seconds", duration)
        return duration

    except Exception as exc:
        logger.exception("Failed to detect video duration: %s", exc)
        return None


def ffprobe_media_info(ffprobe_path: Path, input_video: Path, logger) -> dict:
    show_entries = (
        "format=filename,duration,size,bit_rate:"
        "stream=index,codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,duration,bit_rate:"
        "stream_disposition=attached_pic,default:"
        "stream_tags=handler_name,comment,creation_time,language"
    )
    cmd = [
        str(ffprobe_path),
        "-v", "error",
        "-show_entries", show_entries,
        "-of", "json",
        str(input_video),
    ]
    result = run_command_capture(cmd, logger)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {input_video}: {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse ffprobe JSON for {input_video}: {exc}") from exc


def parse_ffprobe_rate(raw_value: str | None) -> float | None:
    if not raw_value or raw_value in {"0/0", "N/A"}:
        return None
    if "/" in raw_value:
        numerator, denominator = raw_value.split("/", 1)
        try:
            num = float(numerator)
            den = float(denominator)
            if den == 0:
                return None
            return num / den
        except ValueError:
            return None
    try:
        return float(raw_value)
    except ValueError:
        return None


def is_real_video_stream(stream_payload: dict) -> bool:
    if (stream_payload.get("codec_type") or "").lower() != "video":
        return False

    disposition = stream_payload.get("disposition") or {}
    if disposition.get("attached_pic") == 1:
        return False

    width = int(stream_payload.get("width") or 0)
    height = int(stream_payload.get("height") or 0)
    return width > 0 and height > 0


def summarize_streams(ffprobe_payload: dict, logger) -> tuple[list[dict], list[dict]]:
    streams = ffprobe_payload.get("streams") or []
    all_video_streams: list[dict] = []
    candidate_streams: list[dict] = []

    for stream_payload in streams:
        if (stream_payload.get("codec_type") or "").lower() != "video":
            continue

        tags = stream_payload.get("tags") or {}
        disposition = stream_payload.get("disposition") or {}
        stream_summary = {
            "stream_index": stream_payload.get("index"),
            "video_ordinal": len(all_video_streams),
            "codec_name": stream_payload.get("codec_name"),
            "width": stream_payload.get("width"),
            "height": stream_payload.get("height"),
            "avg_frame_rate": parse_ffprobe_rate(stream_payload.get("avg_frame_rate")),
            "r_frame_rate": parse_ffprobe_rate(stream_payload.get("r_frame_rate")),
            "duration": float(stream_payload["duration"]) if stream_payload.get("duration") not in (None, "N/A") else None,
            "bit_rate": int(stream_payload["bit_rate"]) if stream_payload.get("bit_rate") not in (None, "N/A") else None,
            "handler_name": tags.get("handler_name"),
            "comment": tags.get("comment"),
            "language": tags.get("language"),
            "attached_pic": bool(disposition.get("attached_pic", 0)),
            "raw": stream_payload,
        }
        all_video_streams.append(stream_summary)
        if is_real_video_stream(stream_payload):
            candidate_streams.append(stream_summary)

    logger.info("Video streams detected: %s", len(all_video_streams))
    logger.info("Candidate non-attached video streams: %s", len(candidate_streams))
    for stream in candidate_streams:
        logger.info(
            "Candidate stream idx=%s ordinal=%s size=%sx%s codec=%s attached_pic=%s handler=%s comment=%s",
            stream["stream_index"],
            stream["video_ordinal"],
            stream["width"],
            stream["height"],
            stream["codec_name"],
            stream["attached_pic"],
            stream["handler_name"],
            stream["comment"],
        )

    excluded = [stream for stream in all_video_streams if stream not in candidate_streams]
    for stream in excluded:
        logger.info(
            "Excluded video stream idx=%s size=%sx%s codec=%s avg_frame_rate=%s r_frame_rate=%s handler=%s",
            stream["stream_index"],
            stream["width"],
            stream["height"],
            stream["codec_name"],
            stream["raw"].get("avg_frame_rate"),
            stream["raw"].get("r_frame_rate"),
            stream["handler_name"],
        )

    return all_video_streams, candidate_streams


def build_extract_command(
    ffmpeg_path: Path,
    input_video: Path,
    output_pattern: Path,
    fps: float | None,
    jpg_quality: int,
    overwrite: bool,
    stream_index: int | None,
) -> list[str]:
    cmd = [str(ffmpeg_path)]
    cmd.append("-y" if overwrite else "-n")
    cmd.extend(["-i", str(input_video)])

    if stream_index is not None:
        cmd.extend(["-map", f"0:{stream_index}"])

    vf_parts: list[str] = []
    if fps is not None:
        vf_parts.append(f"fps={fps}")
    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])

    cmd.extend(["-q:v", str(jpg_quality), str(output_pattern)])
    return cmd


def clear_existing_outputs(paths: dict[str, Path], prefix: str, logger) -> int:
    removed = 0
    for file_path in paths["frames_360"].glob(f"{prefix}_*.jpg"):
        file_path.unlink()
        removed += 1
        logger.debug("Deleted old flat frame: %s", file_path)

    streams_root = paths["frames_360_streams"]
    if streams_root.exists():
        shutil.rmtree(streams_root)
        removed += 1
        logger.debug("Deleted old multistream directory: %s", streams_root)
        streams_root.mkdir(parents=True, exist_ok=True)

    if paths["extraction_metadata"].exists():
        paths["extraction_metadata"].unlink()
        removed += 1
        logger.debug("Deleted old extraction metadata: %s", paths["extraction_metadata"])

    return removed


def find_default_input_video(input_dir: Path) -> Path:
    videos = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )

    if not videos:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise FileNotFoundError(
            f"No supported video files found in {input_dir}. "
            f"Supported extensions: {supported}. "
            "Place your source video there or pass --input."
        )

    return videos[0]


def resolve_input_video(input_arg: str | None, workspace_root: Path, input_dir: Path) -> Path:
    if not input_arg:
        return find_default_input_video(input_dir)

    raw = Path(input_arg).expanduser()
    candidate_bases: list[Path] = []
    if raw.is_absolute():
        candidate_bases.append(raw)
    else:
        candidate_bases.extend([
            raw,
            Path.cwd() / raw,
            workspace_root / raw,
            input_dir / raw,
        ])

    seen: set[Path] = set()
    candidates: list[Path] = []
    for base in candidate_bases:
        resolved = base.resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)
        if resolved.suffix == "":
            for ext in sorted(SUPPORTED_VIDEO_EXTENSIONS):
                with_ext = resolved.with_suffix(ext)
                if with_ext not in seen:
                    seen.add(with_ext)
                    candidates.append(with_ext)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    return candidates[0]


def load_json_file(path: Path, logger) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse JSON file %s: %s", path, exc)
        return None


def select_streams(
    args,
    candidate_streams: list[dict],
    preprocess_payload: dict | None,
    logger,
) -> tuple[list[dict], str, dict | None]:
    if not candidate_streams:
        raise RuntimeError("No usable non-attached video streams were found in the input video.")

    selected: list[dict] | None = None
    mode = "default_first_candidate"
    preprocess_recommendation = None

    if args.video_stream_index is not None:
        selected = [stream for stream in candidate_streams if stream["stream_index"] == args.video_stream_index]
        if not selected:
            available = [stream["stream_index"] for stream in candidate_streams]
            raise RuntimeError(
                f"Requested --video-stream-index {args.video_stream_index} is not a usable candidate stream. Available candidate streams: {available}"
            )
        mode = "explicit_single_stream"
        return selected, mode, preprocess_recommendation

    if args.extract_all_real_video_streams:
        return list(candidate_streams), "explicit_all_candidate_streams", preprocess_recommendation

    if args.use_preprocess_recommendation and preprocess_payload:
        preprocess_recommendation = preprocess_payload.get("recommendation") or {}
        strategy = preprocess_recommendation.get("recommended_strategy")
        primary_stream_index = preprocess_recommendation.get("recommended_primary_stream_index")

        if strategy == "extract_both_then_stitch":
            logger.info("Using preprocess recommendation: extract_both_then_stitch -> selecting all candidate streams")
            return list(candidate_streams), "preprocess_recommendation_all_candidate_streams", preprocess_recommendation

        if primary_stream_index is not None:
            selected = [stream for stream in candidate_streams if stream["stream_index"] == primary_stream_index]
            if selected:
                logger.info("Using preprocess recommendation primary stream index: %s", primary_stream_index)
                return selected, "preprocess_recommendation_primary_stream", preprocess_recommendation
            logger.warning(
                "Preprocess recommendation requested primary stream %s, but it is not present among current candidate streams.",
                primary_stream_index,
            )

        logger.info("Preprocess recommendation did not resolve to a concrete stream selection; falling back to default first candidate stream")
        return [candidate_streams[0]], "preprocess_recommendation_fallback_first_candidate", preprocess_recommendation

    return [candidate_streams[0]], mode, preprocess_recommendation


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png"}
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 360 frames from a video into data/frames_360. Supports either a single selected video stream or all real candidate video streams into paired subfolders."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input video. Defaults to first supported file in data/input_video.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frames per second to extract. Example: 2.0",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=None,
        help="Approximate total number of frames to extract from the whole video.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame360",
        help="Filename prefix for extracted frames.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=2,
        help="JPEG quality for FFmpeg q:v. Lower is better. Typical 2 to 5.",
    )
    parser.add_argument(
        "--video-stream-index",
        type=int,
        default=None,
        help="Explicitly extract only one video stream index, for example 0 or 1.",
    )
    parser.add_argument(
        "--extract-all-real-video-streams",
        action="store_true",
        help="Extract every non-attached candidate video stream into data/frames_360/streams/stream_XX/.",
    )
    parser.add_argument(
        "--use-preprocess-recommendation",
        action="store_true",
        help="Read data/input_video/_preprocess_metadata.json and follow its recommended stream selection when possible. Explicit stream flags still override this.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing matching output files.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing extracted frames and extraction metadata before extraction.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose console logging and subprocess line streaming.",
    )

    args = parser.parse_args()

    if args.fps is not None and args.target_frames is not None:
        parser.error("Use either --fps or --target-frames, not both.")

    if args.fps is None and args.target_frames is None:
        args.fps = 2.0

    return args


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

        ffmpeg_path = find_ffmpeg(code_root)
        logger.info("FFmpeg found: %s", ffmpeg_path)

        ffprobe_path = find_ffprobe(ffmpeg_path)
        if ffprobe_path:
            logger.info("ffprobe found: %s", ffprobe_path)
        else:
            logger.warning("ffprobe was not found; target-frame mode and multistream probing will not work")

        input_video = resolve_input_video(args.input, workspace_root, paths["input_video"])
        if not input_video.exists():
            logger.error("Input video not found: %s", input_video)
            return 1
        logger.info("Input video: %s", input_video)

        if ffprobe_path is None and (args.target_frames is not None or args.extract_all_real_video_streams or args.video_stream_index is not None or args.use_preprocess_recommendation):
            logger.error("ffprobe is required for target-frame mode and stream-aware extraction options")
            return 1

        fps = args.fps
        duration = None
        if args.target_frames is not None:
            duration = ffprobe_duration(ffprobe_path, input_video, logger) if ffprobe_path else None
            if duration is None or duration <= 0:
                logger.error("Could not determine video duration with ffprobe")
                return 1
            fps = max(args.target_frames / duration, 0.01)
            logger.info(
                "Computed FPS from target frame count: target=%s duration=%.2fs fps=%.6f",
                args.target_frames,
                duration,
                fps,
            )

        ffprobe_payload = ffprobe_media_info(ffprobe_path, input_video, logger) if ffprobe_path else None
        all_video_streams, candidate_streams = summarize_streams(ffprobe_payload, logger) if ffprobe_payload else ([], [])

        preprocess_payload = load_json_file(paths["preprocess_metadata"], logger) if args.use_preprocess_recommendation else None
        selected_streams, extraction_mode, preprocess_recommendation = select_streams(
            args=args,
            candidate_streams=candidate_streams,
            preprocess_payload=preprocess_payload,
            logger=logger,
        )

        output_layout = "flat" if len(selected_streams) == 1 else "streams"
        logger.info("Selected extraction mode: %s", extraction_mode)
        logger.info("Selected output layout: %s", output_layout)
        logger.info("Selected stream indexes: %s", [stream["stream_index"] for stream in selected_streams])

        if args.clean:
            removed = clear_existing_outputs(paths, args.prefix, logger)
            logger.info("Removed %s previous extraction artifact(s)", removed)

        extracted_stream_summaries: list[dict] = []

        for stream in selected_streams:
            stream_index = int(stream["stream_index"])
            if output_layout == "flat":
                output_dir = paths["frames_360"]
                output_pattern = output_dir / f"{args.prefix}_%04d.jpg"
            else:
                output_dir = paths["frames_360_streams"] / f"stream_{stream_index:02d}"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_pattern = output_dir / f"{args.prefix}_%04d.jpg"

            logger.info("Output directory for stream %s: %s", stream_index, output_dir)
            logger.info("Output pattern for stream %s: %s", stream_index, output_pattern)

            cmd = build_extract_command(
                ffmpeg_path=ffmpeg_path,
                input_video=input_video,
                output_pattern=output_pattern,
                fps=fps,
                jpg_quality=args.quality,
                overwrite=args.overwrite,
                stream_index=stream_index,
            )

            run_command_streaming(cmd, logger, verbose=args.verbose)
            frame_count = count_images(output_dir)
            logger.info("Extracted %s frame(s) for stream %s", frame_count, stream_index)

            extracted_stream_summaries.append(
                {
                    "stream_index": stream_index,
                    "video_ordinal": stream.get("video_ordinal"),
                    "codec_name": stream.get("codec_name"),
                    "width": stream.get("width"),
                    "height": stream.get("height"),
                    "avg_frame_rate": stream.get("avg_frame_rate"),
                    "r_frame_rate": stream.get("r_frame_rate"),
                    "duration": stream.get("duration"),
                    "bit_rate": stream.get("bit_rate"),
                    "output_dir": str(output_dir),
                    "output_pattern": str(output_pattern),
                    "frame_count": frame_count,
                }
            )

        extraction_payload = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_script": SCRIPT_NAME,
            "input_video": str(input_video),
            "input_video_name": input_video.name,
            "frame_prefix": args.prefix,
            "jpg_quality": args.quality,
            "fps": fps,
            "target_frames": args.target_frames,
            "output_layout": output_layout,
            "extraction_mode": extraction_mode,
            "used_preprocess_recommendation": bool(args.use_preprocess_recommendation),
            "preprocess_metadata_path": str(paths["preprocess_metadata"]),
            "preprocess_recommendation": preprocess_recommendation,
            "video_streams_all": all_video_streams,
            "candidate_stream_indexes": [stream["stream_index"] for stream in candidate_streams],
            "selected_streams": extracted_stream_summaries,
            "paths": {
                "frames_360": str(paths["frames_360"]),
                "frames_360_streams": str(paths["frames_360_streams"]),
                "metadata": str(paths["extraction_metadata"]),
            },
            "notes": [
                "Single-stream extraction writes frames directly into data/frames_360.",
                "Multi-stream extraction writes frames into data/frames_360/streams/stream_XX so paired downstream normalization or stitching can consume them later.",
            ],
            "tools": {
                "ffmpeg_path": str(ffmpeg_path),
                "ffprobe_path": str(ffprobe_path) if ffprobe_path else None,
            },
        }
        paths["extraction_metadata"].write_text(json.dumps(extraction_payload, indent=2), encoding="utf-8")
        logger.info("Wrote extraction metadata: %s", paths["extraction_metadata"])

        total_frames = sum(item["frame_count"] for item in extracted_stream_summaries)
        logger.info("Extraction completed successfully")
        logger.info("Total extracted frame(s): %s", total_frames)

        print("\n=== Extract Frames Report ===")
        print(f"Input video: {input_video.name}")
        print(f"Extraction mode: {extraction_mode}")
        print(f"Output layout: {output_layout}")
        print(f"Selected streams: {[item['stream_index'] for item in extracted_stream_summaries]}")
        for item in extracted_stream_summaries:
            print(
                f"  - stream {item['stream_index']}: {item['frame_count']} frame(s) -> {item['output_dir']}"
            )
        print(f"Metadata: {paths['extraction_metadata']}")

        return 0

    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
