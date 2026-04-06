from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Make ./scripts importable when running:
# python .\scripts\pipeline.py
sys.path.append(str(Path(__file__).resolve().parent))

from common.logging_utils import setup_logger
from common.presets import DEFAULT_PRESET_NAME, get_preset_names


SCRIPT_NAME = "pipeline"

STEP_ORDER = [
    "extract_frames",
    "convert_360_to_views",
    "prepare_colmap_images",
    "run_colmap",
]

WARNING_PATTERNS = [
    re.compile(r"\bwarning\b", re.IGNORECASE),
    re.compile(r"\bwarn\b", re.IGNORECASE),
]

ERROR_PATTERNS = [
    re.compile(r"\berror\b", re.IGNORECASE),
    re.compile(r"\bfatal\b", re.IGNORECASE),
    re.compile(r"\bexception\b", re.IGNORECASE),
]


@dataclass
class StepResult:
    step_name: str
    command: list[str]
    exit_code: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_dirs(root: Path, logger) -> dict[str, Path]:
    paths = {
        "root": root,
        "data": root / "data",
        "input_video": root / "data" / "input_video",
        "frames_360": root / "data" / "frames_360",
        "frames_perspective": root / "data" / "frames_perspective",
        "colmap": root / "data" / "colmap",
        "colmap_images": root / "data" / "colmap" / "images",
        "colmap_sparse": root / "data" / "colmap" / "sparse",
        "logs": root / "logs",
        "scripts": root / "scripts",
        "extract_script": root / "scripts" / "extract_frames.py",
        "convert_script": root / "scripts" / "convert_360_to_views.py",
        "prepare_script": root / "scripts" / "prepare_colmap_images.py",
        "colmap_script": root / "scripts" / "run_colmap.py",
        "report_script": root / "scripts" / "pipeline_report.py",
    }

    for key in (
        "data",
        "input_video",
        "frames_360",
        "frames_perspective",
        "colmap",
        "colmap_images",
        "colmap_sparse",
        "logs",
        "scripts",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s -> %s", key, paths[key])

    return paths


def quote_cmd(cmd: Iterable[str]) -> str:
    return " ".join(f'"{c}"' if " " in c else c for c in cmd)


def line_matches_any(line: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(line) for p in patterns)


def stream_process_output(
    process: subprocess.Popen,
    logger,
    prefix: str,
    verbose: bool,
    result: StepResult | None = None,
) -> None:
    assert process.stdout is not None

    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if not line:
            continue

        logger.debug("%s %s", prefix, line)

        if result is not None:
            if line_matches_any(line, WARNING_PATTERNS):
                result.warnings.append(line)
            if line_matches_any(line, ERROR_PATTERNS):
                result.errors.append(line)

        if verbose:
            print(f"{prefix} {line}")


def run_command_streaming(
    cmd: list[str],
    logger,
    verbose: bool = False,
    cwd: Path | None = None,
    step_name: str = "unknown",
) -> StepResult:
    logger.info("Running command")
    logger.debug("Command: %s", quote_cmd(cmd))
    if cwd is not None:
        logger.debug("Working directory: %s", cwd)

    result = StepResult(step_name=step_name, command=cmd)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        shell=False,
        cwd=str(cwd) if cwd else None,
    )

    try:
        stream_process_output(
            process,
            logger,
            prefix="[SUBPROCESS]",
            verbose=verbose,
            result=result,
        )
    finally:
        return_code = process.wait()

    result.exit_code = return_code
    logger.debug("Command exit code: %s", return_code)

    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}")

    return result


def validate_step_name(step_name: str) -> str:
    if step_name not in STEP_ORDER:
        raise ValueError(f"Invalid step '{step_name}'. Valid steps: {STEP_ORDER}")
    return step_name


def steps_to_run(step_from: str, step_to: str) -> list[str]:
    start_idx = STEP_ORDER.index(step_from)
    end_idx = STEP_ORDER.index(step_to)
    if start_idx > end_idx:
        raise ValueError(f"step-from '{step_from}' must not come after step-to '{step_to}'")
    return STEP_ORDER[start_idx : end_idx + 1]


def append_flag(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def append_optional_value(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png"}
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def build_extract_frames_cmd(paths: dict[str, Path], args, python_exe: str) -> list[str]:
    cmd = [python_exe, str(paths["extract_script"])]

    append_optional_value(cmd, "--input", args.input_video)
    append_optional_value(cmd, "--prefix", args.frame_prefix)
    append_optional_value(cmd, "--quality", args.frame_quality)

    if args.extract_fps is not None:
        append_optional_value(cmd, "--fps", args.extract_fps)
    elif args.target_frames is not None:
        append_optional_value(cmd, "--target-frames", args.target_frames)

    append_flag(cmd, "--overwrite", args.overwrite)
    append_flag(cmd, "--clean", args.clean_extract)
    append_flag(cmd, "--verbose", args.verbose)

    return cmd


def build_convert_views_cmd(paths: dict[str, Path], args, python_exe: str) -> list[str]:
    cmd = [
        python_exe,
        str(paths["convert_script"]),
        "--preset",
        args.preset,
    ]

    append_optional_value(cmd, "--input-prefix", args.frame_prefix)

    if args.views:
        cmd.extend(["--views", *args.views])

    append_optional_value(cmd, "--h-fov", args.h_fov)
    append_optional_value(cmd, "--v-fov", args.v_fov)
    append_optional_value(cmd, "--width", args.view_width)
    append_optional_value(cmd, "--height", args.view_height)
    append_optional_value(cmd, "--pitch", args.pitch)
    append_optional_value(cmd, "--roll", args.roll)
    append_optional_value(cmd, "--interpolation", args.interpolation)
    append_optional_value(cmd, "--quality", args.view_quality)
    append_optional_value(cmd, "--limit", args.convert_limit)

    append_flag(cmd, "--overwrite", args.overwrite)
    append_flag(cmd, "--clean", args.clean_convert)
    append_flag(cmd, "--verbose", args.verbose)

    return cmd


def build_prepare_colmap_images_cmd(paths: dict[str, Path], args, python_exe: str) -> list[str]:
    cmd = [python_exe, str(paths["prepare_script"])]

    if args.views:
        cmd.extend(["--views", *args.views])

    append_optional_value(cmd, "--input-prefix", args.frame_prefix)
    append_optional_value(cmd, "--copy-mode", args.copy_mode)
    append_optional_value(cmd, "--limit", args.prepare_limit)

    append_flag(cmd, "--clean", args.clean_prepare)
    append_flag(cmd, "--strict", args.strict_prepare)
    append_flag(cmd, "--verbose", args.verbose)

    return cmd


def build_run_colmap_cmd(paths: dict[str, Path], args, python_exe: str) -> list[str]:
    cmd = [
        python_exe,
        str(paths["colmap_script"]),
        "--preset",
        args.preset,
    ]

    append_optional_value(cmd, "--images", args.colmap_images)
    append_optional_value(cmd, "--database", args.colmap_database)
    append_optional_value(cmd, "--sparse", args.colmap_sparse)
    append_optional_value(cmd, "--matcher", args.matcher)
    append_optional_value(cmd, "--camera-model", args.camera_model)
    append_optional_value(cmd, "--max-image-size", args.max_image_size)

    append_flag(cmd, "--single-camera", args.single_camera)
    append_flag(cmd, "--use-gpu", args.use_gpu)
    append_flag(cmd, "--force-cli", args.force_cli)
    append_flag(cmd, "--reset", args.reset_colmap)
    append_flag(cmd, "--verbose", args.verbose)

    return cmd


def validate_extract_outputs(paths: dict[str, Path], args) -> tuple[bool, str]:
    count = len(list(paths["frames_360"].glob(f"{args.frame_prefix}_*.jpg")))
    if count == 0:
        return False, "No extracted 360 frames were found."
    return True, f"Extracted frame count looks valid: {count}"


def validate_convert_outputs(paths: dict[str, Path], args) -> tuple[bool, str]:
    base = paths["frames_perspective"]
    total = 0
    if args.views:
        target_views = args.views
    else:
        target_views = None

    if target_views is None:
        subdirs = [p for p in base.iterdir() if p.is_dir()]
    else:
        subdirs = [base / v for v in target_views]

    for folder in subdirs:
        total += count_images(folder)

    if total == 0:
        return False, "No perspective view images were found."
    return True, f"Perspective image count looks valid: {total}"


def validate_prepare_outputs(paths: dict[str, Path], args) -> tuple[bool, str]:
    count = count_images(paths["colmap_images"])
    if count == 0:
        return False, "COLMAP images folder is empty."
    return True, f"COLMAP image count looks valid: {count}"


def validate_colmap_outputs(paths: dict[str, Path], args) -> tuple[bool, str]:
    db_path = Path(args.colmap_database).resolve() if args.colmap_database else (paths["colmap"] / "database.db")
    sparse_path = Path(args.colmap_sparse).resolve() if args.colmap_sparse else paths["colmap_sparse"]

    model_dirs = [p for p in sparse_path.iterdir() if p.is_dir()] if sparse_path.exists() else []
    if not db_path.exists():
        return False, "COLMAP database.db was not created."
    if len(model_dirs) == 0:
        return False, "No sparse model folders were created."
    return True, f"COLMAP outputs look valid: database present, sparse model folders={len(model_dirs)}"


STEP_VALIDATORS = {
    "extract_frames": validate_extract_outputs,
    "convert_360_to_views": validate_convert_outputs,
    "prepare_colmap_images": validate_prepare_outputs,
    "run_colmap": validate_colmap_outputs,
}


def run_validation(step_name: str, paths: dict[str, Path], args, logger) -> None:
    validator = STEP_VALIDATORS.get(step_name)
    if validator is None:
        return

    ok, message = validator(paths, args)
    if ok:
        logger.info("Validation passed for %s: %s", step_name, message)
    else:
        raise RuntimeError(f"Validation failed for {step_name}: {message}")


def print_pipeline_summary(results: list[StepResult], logger) -> None:
    logger.info("Pipeline step summary:")
    for result in results:
        logger.info(
            "  %s | exit=%s | warnings=%s | errors_detected=%s",
            result.step_name,
            result.exit_code,
            len(result.warnings),
            len(result.errors),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Gaussian-splats preprocessing pipeline end-to-end from video to COLMAP sparse reconstruction."
    )

    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET_NAME,
        choices=get_preset_names(),
        help="Pipeline preset name.",
    )

    parser.add_argument(
        "--step-from",
        type=str,
        default=STEP_ORDER[0],
        help=f"Starting step. One of: {', '.join(STEP_ORDER)}",
    )
    parser.add_argument(
        "--step-to",
        type=str,
        default=STEP_ORDER[-1],
        help=f"Final step. One of: {', '.join(STEP_ORDER)}",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    # New controls
    parser.add_argument(
        "--validation-mode",
        action="store_true",
        help="Run filesystem validation checks after each pipeline step.",
    )
    parser.add_argument(
        "--stop-on-warning",
        action="store_true",
        help="Stop the pipeline if any subprocess warning is detected in a step.",
    )

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--frame-prefix", type=str, default="frame360")

    # Step 1
    parser.add_argument("--input-video", type=str, default=None)
    parser.add_argument("--extract-fps", type=float, default=None)
    parser.add_argument("--target-frames", type=int, default=100)
    parser.add_argument("--frame-quality", type=int, default=2)
    parser.add_argument("--clean-extract", action="store_true")

    # Step 2
    parser.add_argument("--views", nargs="+", default=None)
    parser.add_argument("--h-fov", type=float, default=None)
    parser.add_argument("--v-fov", type=float, default=None)
    parser.add_argument("--view-width", type=int, default=None)
    parser.add_argument("--view-height", type=int, default=None)
    parser.add_argument("--pitch", type=float, default=None)
    parser.add_argument("--roll", type=float, default=None)
    parser.add_argument("--interpolation", type=str, default=None)
    parser.add_argument("--view-quality", type=int, default=None)
    parser.add_argument("--convert-limit", type=int, default=None)
    parser.add_argument("--clean-convert", action="store_true")

    # Step 3
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default="copy")
    parser.add_argument("--prepare-limit", type=int, default=None)
    parser.add_argument("--strict-prepare", action="store_true")
    parser.add_argument("--clean-prepare", action="store_true")

    # Step 4
    parser.add_argument("--colmap-images", type=str, default=None)
    parser.add_argument("--colmap-database", type=str, default=None)
    parser.add_argument("--colmap-sparse", type=str, default=None)
    parser.add_argument("--matcher", type=str, default=None)
    parser.add_argument("--camera-model", type=str, default=None)
    parser.add_argument("--single-camera", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--max-image-size", type=int, default=None)
    parser.add_argument("--force-cli", action="store_true")
    parser.add_argument("--reset-colmap", action="store_true")

    args = parser.parse_args()

    args.step_from = validate_step_name(args.step_from)
    args.step_to = validate_step_name(args.step_to)

    if args.extract_fps is not None and args.target_frames is not None:
        parser.error("Use either --extract-fps or --target-frames, not both.")

    return args


def main() -> int:
    args = parse_args()
    logger, run_log_path, latest_log_path = setup_logger(SCRIPT_NAME, verbose=args.verbose)

    try:
        root = project_root_from_script()
        logger.info("Project root: %s", root)
        logger.info("Run log: %s", run_log_path)
        logger.info("Latest log: %s", latest_log_path)

        paths = ensure_dirs(root, logger)
        logger.info("Project folders checked and created if missing")
        logger.info("Preset: %s", args.preset)

        selected_steps = steps_to_run(args.step_from, args.step_to)
        logger.info("Steps to run: %s", selected_steps)

        python_exe = sys.executable
        logger.info("Python executable: %s", python_exe)

        commands: list[tuple[str, list[str]]] = []

        if "extract_frames" in selected_steps:
            commands.append(("extract_frames", build_extract_frames_cmd(paths, args, python_exe)))
        if "convert_360_to_views" in selected_steps:
            commands.append(("convert_360_to_views", build_convert_views_cmd(paths, args, python_exe)))
        if "prepare_colmap_images" in selected_steps:
            commands.append(("prepare_colmap_images", build_prepare_colmap_images_cmd(paths, args, python_exe)))
        if "run_colmap" in selected_steps:
            commands.append(("run_colmap", build_run_colmap_cmd(paths, args, python_exe)))

        logger.info("Prepared %s pipeline command(s)", len(commands))

        for step_name, cmd in commands:
            logger.info("Pipeline step prepared: %s", step_name)
            logger.debug("Step command: %s", quote_cmd(cmd))

        if args.dry_run:
            logger.info("Dry run enabled; no commands will be executed")
            for step_name, cmd in commands:
                print(f"\n[{step_name}]")
                print(quote_cmd(cmd))
            return 0

        results: list[StepResult] = []

        for index, (step_name, cmd) in enumerate(commands, start=1):
            logger.info("Starting pipeline step %s / %s: %s", index, len(commands), step_name)
            result = run_command_streaming(cmd, logger, verbose=args.verbose, cwd=root, step_name=step_name)
            results.append(result)
            logger.info("Completed pipeline step: %s", step_name)

            if args.stop_on_warning and result.warnings:
                logger.warning("Warnings detected in step %s:", step_name)
                for warning_line in result.warnings[:20]:
                    logger.warning("  %s", warning_line)
                raise RuntimeError(f"Pipeline stopped because warnings were detected in step: {step_name}")

            if args.validation_mode:
                run_validation(step_name, paths, args, logger)

        print_pipeline_summary(results, logger)
        logger.info("Pipeline completed successfully")
        return 0

    except Exception as exc:
        logger.exception("Fatal pipeline error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())