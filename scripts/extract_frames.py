from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
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
        "frames_perspective": workspace_root / "data" / "frames_perspective",
        "colmap": workspace_root / "data" / "colmap",
        "logs": workspace_root / "logs",
        "tools": code_root / "tools",
        "scripts": code_root / "scripts",
    }

    for key in ("data", "input_video", "frames_360", "frames_perspective", "colmap", "logs"):
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


def find_ffprobe(ffmpeg_path: Path) -> Path | None:
    local_ffprobe = ffmpeg_path.parent / "ffprobe.exe"
    if local_ffprobe.exists():
        return local_ffprobe

    ffprobe_on_path = shutil.which("ffprobe")
    if ffprobe_on_path:
        return Path(ffprobe_on_path)

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
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        if result.stdout:
            logger.debug("[FFPROBE STDOUT] %s", result.stdout.strip())
        if result.stderr:
            logger.debug("[FFPROBE STDERR] %s", result.stderr.strip())

        if result.returncode != 0:
            logger.error("ffprobe failed with exit code %s", result.returncode)
            return None

        duration = float(result.stdout.strip())
        logger.debug("Detected duration: %.4f seconds", duration)
        return duration

    except Exception as exc:
        logger.exception("Failed to detect video duration: %s", exc)
        return None


def build_extract_command(
    ffmpeg_path: Path,
    input_video: Path,
    output_pattern: Path,
    fps: float | None,
    jpg_quality: int,
    overwrite: bool,
) -> list[str]:
    cmd = [str(ffmpeg_path)]

    cmd.append("-y" if overwrite else "-n")
    cmd.extend(["-i", str(input_video)])

    vf_parts: list[str] = []
    if fps is not None:
        vf_parts.append(f"fps={fps}")

    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])

    cmd.extend(
        [
            "-q:v",
            str(jpg_quality),
            str(output_pattern),
        ]
    )

    return cmd


def clear_existing_frames(output_dir: Path, prefix: str, logger) -> int:
    removed = 0
    for file_path in output_dir.glob(f"{prefix}_*.jpg"):
        file_path.unlink()
        removed += 1
        logger.debug("Deleted old frame: %s", file_path)

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


def list_supported_videos(input_dir: Path) -> list[Path]:
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )


def resolve_input_video(
    input_arg: str,
    workspace_root: Path,
    input_dir: Path,
) -> Path:
    raw = Path(input_arg).expanduser()

    candidate_bases: list[Path] = []

    if raw.is_absolute():
        candidate_bases.append(raw)
    else:
        candidate_bases.extend(
            [
                raw,
                Path.cwd() / raw,
                workspace_root / raw,
                input_dir / raw,
            ]
        )

    seen: set[Path] = set()
    candidates: list[Path] = []

    for base in candidate_bases:
        try:
            resolved = base.resolve(strict=False)
        except Exception:
            resolved = base

        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)

        # If the caller omitted the extension, try known video extensions.
        if resolved.suffix == "":
            for ext in sorted(SUPPORTED_VIDEO_EXTENSIONS):
                with_ext = resolved.with_suffix(ext)
                if with_ext not in seen:
                    seen.add(with_ext)
                    candidates.append(with_ext)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    # Return the most sensible normalized path for error reporting.
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 360 frames from a video into data/frames_360."
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing matching output files.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing extracted frames with the same prefix before extraction.",
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
            logger.warning("ffprobe was not found; target-frame mode will not work")

        available_videos = list_supported_videos(paths["input_video"])

        if args.input:
            input_video = resolve_input_video(
                input_arg=args.input,
                workspace_root=workspace_root,
                input_dir=paths["input_video"],
            )
        else:
            input_video = find_default_input_video(paths["input_video"])

        if not input_video.exists():
            logger.error("Input video not found: %s", input_video)

            if available_videos:
                logger.info("Supported videos currently found in %s:", paths["input_video"])
                for video_path in available_videos:
                    logger.info(" - %s", video_path.name)
            else:
                logger.info(
                    "No supported videos found in %s. Supported extensions: %s",
                    paths["input_video"],
                    ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS)),
                )

            return 1

        logger.info("Input video: %s", input_video)

        fps = args.fps
        if args.target_frames is not None:
            if ffprobe_path is None:
                logger.error("target-frames was requested but ffprobe is not available")
                return 1

            duration = ffprobe_duration(ffprobe_path, input_video, logger)
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

        output_dir = paths["frames_360"]
        output_pattern = output_dir / f"{args.prefix}_%04d.jpg"

        logger.info("Output directory: %s", output_dir)
        logger.info("Output pattern: %s", output_pattern)

        if args.clean:
            removed = clear_existing_frames(output_dir, args.prefix, logger)
            logger.info("Removed %s old frame(s) with prefix '%s'", removed, args.prefix)

        cmd = build_extract_command(
            ffmpeg_path=ffmpeg_path,
            input_video=input_video,
            output_pattern=output_pattern,
            fps=fps,
            jpg_quality=args.quality,
            overwrite=args.overwrite,
        )

        run_command_streaming(cmd, logger, verbose=args.verbose)

        extracted_files = sorted(output_dir.glob(f"{args.prefix}_*.jpg"))
        logger.info("Extraction completed successfully")
        logger.info("Extracted %s frame(s)", len(extracted_files))

        return 0

    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
