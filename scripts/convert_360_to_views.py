from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

sys.path.append(str(Path(__file__).resolve().parent))

from common.logging_utils import setup_logger
from common.workspace import resolve_code_root, resolve_workspace_root
from common.presets import DEFAULT_PRESET_NAME, get_preset, get_preset_names
from common.config_merge import deep_merge_dict, build_projection_override_dict


SCRIPT_NAME = "convert_360_to_views"


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

    raise FileNotFoundError(
        "FFmpeg was not found. Expected one of:\n"
        f"  - {code_root / 'tools' / 'ffmpeg' / 'bin' / 'ffmpeg.exe'}\n"
        f"  - {code_root / 'tools' / 'ffmpeg.exe'}"
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
    yaw: float,
    h_fov: float,
    v_fov: float,
    out_width: int,
    out_height: int,
    pitch: float,
    roll: float,
    interpolation: str,
    overwrite: bool,
    jpg_quality: int,
) -> list[str]:
    vf = (
        f"v360=input=equirect:output=flat:"
        f"yaw={yaw}:pitch={pitch}:roll={roll}:"
        f"h_fov={h_fov}:v_fov={v_fov}:"
        f"w={out_width}:h={out_height}:"
        f"interp={interpolation}"
    )

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
        description="Convert equirectangular 360 frames into perspective views."
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
        "--views",
        nargs="+",
        default=None,
        help="Override views from preset.",
    )
    parser.add_argument("--h-fov", type=float, default=None)
    parser.add_argument("--v-fov", type=float, default=None)
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

        logger.info("Preset: %s", args.preset)
        logger.info("Found %s input frame(s)", len(input_frames))
        logger.info("Views to generate: %s", views)
        logger.info(
            "Projection settings: h_fov=%.2f v_fov=%.2f width=%s height=%s pitch=%.2f roll=%.2f interp=%s quality=%s",
            projection_config["h_fov"],
            projection_config["v_fov"],
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
                    yaw=yaw,
                    h_fov=projection_config["h_fov"],
                    v_fov=projection_config["v_fov"],
                    out_width=projection_config["width"],
                    out_height=projection_config["height"],
                    pitch=projection_config["pitch"],
                    roll=projection_config["roll"],
                    interpolation=projection_config["interpolation"],
                    overwrite=args.overwrite,
                    jpg_quality=projection_config["quality"],
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
