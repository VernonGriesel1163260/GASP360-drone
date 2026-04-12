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
from common.workspace import resolve_code_root, resolve_workspace_root
from common.presets import DEFAULT_PRESET_NAME, get_preset, get_preset_names


SCRIPT_NAME = "pipeline"

STEP_ORDER = [
    "preprocess_input_video",
    "extract_frames",
    "normalize_multistream_360",
    "convert_360_to_views",
    "prepare_colmap_images",
    "run_colmap",
]

EXTRACTION_METADATA_FILENAME = "_extraction_metadata.json"
NORMALIZATION_METADATA_FILENAME = "_normalization_metadata.json"


def load_json_file(path: Path) -> dict | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

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
        "colmap_images": workspace_root / "data" / "colmap" / "images",
        "colmap_sparse": workspace_root / "data" / "colmap" / "sparse",
        "logs": workspace_root / "logs",
        "scripts": code_root / "scripts",
        "preprocess_script": code_root / "scripts" / "preprocess_input_video.py",
        "extract_script": code_root / "scripts" / "extract_frames.py",
        "normalize_script": code_root / "scripts" / "normalize_multistream_360.py",
        "convert_script": code_root / "scripts" / "convert_360_to_views.py",
        "prepare_script": code_root / "scripts" / "prepare_colmap_images.py",
        "colmap_script": code_root / "scripts" / "run_colmap.py",
        "report_script": code_root / "scripts" / "pipeline_report.py",
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


def build_preprocess_cmd(paths: dict[str, Path], args, python_exe: str) -> list[str]:
    cmd = [python_exe, str(paths["preprocess_script"])]

    append_optional_value(cmd, "--input", args.input_video)
    append_optional_value(cmd, "--mode", args.preprocess_mode)
    append_optional_value(cmd, "--sample-count", args.preprocess_sample_count)

    if args.preprocess_sample_positions:
        cmd.extend(["--sample-positions", *[str(v) for v in args.preprocess_sample_positions]])

    append_optional_value(cmd, "--force-primary-stream-index", args.preprocess_force_primary_stream_index)
    append_optional_value(cmd, "--force-frame-format", args.preprocess_force_frame_format)
    append_optional_value(cmd, "--force-strategy", args.preprocess_force_strategy)

    append_flag(cmd, "--clean", args.clean_preprocess)
    append_flag(cmd, "--overwrite", args.overwrite)
    append_flag(cmd, "--verbose", args.verbose)

    return cmd


def build_extract_frames_cmd(paths: dict[str, Path], args, python_exe: str) -> list[str]:
    cmd = [python_exe, str(paths["extract_script"])]

    append_optional_value(cmd, "--input", args.input_video)
    append_optional_value(cmd, "--prefix", args.frame_prefix)
    append_optional_value(cmd, "--quality", args.frame_quality)

    if args.extract_fps is not None:
        append_optional_value(cmd, "--fps", args.extract_fps)
    elif args.target_frames is not None:
        append_optional_value(cmd, "--target-frames", args.target_frames)

    append_optional_value(cmd, "--video-stream-index", args.extract_video_stream_index)
    append_flag(cmd, "--extract-all-real-video-streams", args.extract_all_real_video_streams)
    append_flag(cmd, "--use-preprocess-recommendation", args.extract_use_preprocess_recommendation)

    append_flag(cmd, "--overwrite", args.overwrite)
    append_flag(cmd, "--clean", args.clean_extract)
    append_flag(cmd, "--verbose", args.verbose)

    return cmd


def build_normalize_multistream_cmd(paths: dict[str, Path], args, python_exe: str) -> list[str]:
    cmd = [python_exe, str(paths["normalize_script"])]

    append_optional_value(cmd, "--mode", args.normalize_multistream_mode)

    if args.normalize_stream_pair:
        cmd.extend(["--stream-pair", *[str(v) for v in args.normalize_stream_pair]])

    append_flag(cmd, "--use-preprocess-recommendation", args.normalize_use_preprocess_recommendation)
    append_optional_value(cmd, "--output-format", args.normalize_output_format)
    append_optional_value(cmd, "--layout", args.normalize_layout)
    append_optional_value(cmd, "--resize-streams-to", args.normalize_resize_streams_to)
    append_optional_value(cmd, "--rotate-a", args.normalize_rotate_a)
    append_optional_value(cmd, "--rotate-b", args.normalize_rotate_b)

    append_flag(cmd, "--flip-h-a", args.normalize_flip_h_a)
    append_flag(cmd, "--flip-v-a", args.normalize_flip_v_a)
    append_flag(cmd, "--flip-h-b", args.normalize_flip_h_b)
    append_flag(cmd, "--flip-v-b", args.normalize_flip_v_b)

    append_optional_value(cmd, "--output-prefix", args.normalize_output_prefix)
    append_optional_value(cmd, "--limit", args.normalize_limit)

    append_flag(cmd, "--overwrite", args.overwrite)
    append_flag(cmd, "--clean", args.clean_normalize)
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
    input_format_value = args.input_format
    if input_format_value is None and args.normalize_multistream_mode != "off":
        input_format_value = "auto"
    append_optional_value(cmd, "--input-format", input_format_value)
    append_optional_value(cmd, "--input-h-fov", args.input_h_fov)
    append_optional_value(cmd, "--input-v-fov", args.input_v_fov)
    append_optional_value(cmd, "--input-d-fov", args.input_d_fov)

    if args.views:
        cmd.extend(["--views", *args.views])

    append_optional_value(cmd, "--h-fov", args.h_fov)
    append_optional_value(cmd, "--v-fov", args.v_fov)
    append_optional_value(cmd, "--d-fov", args.d_fov)
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

    selected_views = args.views
    if not selected_views:
        selected_views = get_preset(args.preset)["projection"]["views"]
    if selected_views:
        cmd.extend(["--views", *selected_views])

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
    flat_count = len(list(paths["frames_360"].glob(f"{args.frame_prefix}_*.jpg")))
    streams_root = paths["frames_360"] / "streams"
    stream_counts: dict[str, int] = {}
    total_stream_count = 0
    if streams_root.exists():
        for child in sorted(streams_root.iterdir()):
            if child.is_dir():
                count = count_images(child)
                stream_counts[child.name] = count
                total_stream_count += count

    metadata_payload = load_json_file(paths["frames_360"] / EXTRACTION_METADATA_FILENAME)
    layout = metadata_payload.get("output_layout") if metadata_payload else None

    if flat_count == 0 and total_stream_count == 0:
        return False, "No extracted 360 frames were found in flat or multistream output locations."

    if layout == "streams" or total_stream_count > 0:
        return True, f"Multistream extracted frame counts look valid: {stream_counts}"

    return True, f"Extracted frame count looks valid: {flat_count}"


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


def validate_preprocess_outputs(paths: dict[str, Path], args) -> tuple[bool, str]:
    if args.preprocess_mode == "off":
        return True, "Preprocess step disabled."
    metadata_path = paths["input_video"] / "_preprocess_metadata.json"
    if not metadata_path.exists():
        return False, f"Preprocess metadata sidecar was not created: {metadata_path}"
    return True, f"Preprocess metadata sidecar exists: {metadata_path}"


def validate_normalize_outputs(paths: dict[str, Path], args) -> tuple[bool, str]:
    if args.normalize_multistream_mode == "off":
        return True, "Normalize multistream step disabled."
    metadata_path = paths["frames_360"] / NORMALIZATION_METADATA_FILENAME
    if not metadata_path.exists():
        return False, f"Normalization metadata sidecar was not created: {metadata_path}"
    flat_count = len(list(paths["frames_360"].glob(f"{(args.normalize_output_prefix or args.frame_prefix)}_*.jpg")))
    if flat_count == 0:
        return False, "Normalize multistream did not produce any flat normalized frames."
    return True, f"Normalization metadata sidecar exists and produced {flat_count} normalized frame(s)."


STEP_VALIDATORS = {
    "preprocess_input_video": validate_preprocess_outputs,
    "extract_frames": validate_extract_outputs,
    "normalize_multistream_360": validate_normalize_outputs,
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

    # Preprocess step
    parser.add_argument(
        "--preprocess-mode",
        choices=["off", "report-only", "auto"],
        default="off",
        help="Preprocess input container before frame extraction. 'off' skips the step, 'report-only' writes recommendations only, and 'auto' is reserved for future automatic wiring while still writing a report today.",
    )
    parser.add_argument(
        "--preprocess-sample-count",
        type=int,
        default=3,
        help="Number of evenly spaced timestamps to sample per candidate stream during preprocessing.",
    )
    parser.add_argument(
        "--preprocess-sample-positions",
        nargs="*",
        type=float,
        default=None,
        help="Optional sample position fractions for preprocessing, for example 0.2 0.5 0.8.",
    )
    parser.add_argument(
        "--preprocess-force-primary-stream-index",
        type=int,
        default=None,
        help="Force the recommended primary stream index in the preprocess sidecar.",
    )
    parser.add_argument(
        "--preprocess-force-frame-format",
        type=str,
        default=None,
        help="Force the recommended frame format in the preprocess sidecar, for example equirect, fisheye, dfisheye, or flat.",
    )
    parser.add_argument(
        "--preprocess-force-strategy",
        choices=["single_stream", "single_stream_best_of_n", "extract_both_then_stitch", "manual_review_required"],
        default=None,
        help="Force the recommended preprocess strategy in the sidecar.",
    )
    parser.add_argument("--clean-preprocess", action="store_true")

    # Step 1
    parser.add_argument("--input-video", type=str, default=None)
    parser.add_argument("--extract-fps", type=float, default=None)
    parser.add_argument("--target-frames", type=int, default=100)
    parser.add_argument("--frame-quality", type=int, default=2)
    parser.add_argument("--extract-video-stream-index", type=int, default=None, help="Explicitly extract one selected video stream index.")
    parser.add_argument("--extract-all-real-video-streams", action="store_true", help="Extract all non-attached candidate video streams into data/frames_360/streams/stream_XX.")
    parser.add_argument("--extract-use-preprocess-recommendation", action="store_true", help="Let extract_frames follow the preprocess sidecar recommendation when possible. Explicit stream flags still override this.")
    parser.add_argument("--clean-extract", action="store_true")

    # Step 2
    parser.add_argument(
        "--normalize-multistream-mode",
        choices=["off", "auto", "explicit"],
        default="off",
        help="Normalize extracted multistream folders into a flat frame set under data/frames_360. 'off' skips the step, 'auto' infers pair/layout/format, and 'explicit' expects manual overrides where needed.",
    )
    parser.add_argument("--normalize-stream-pair", nargs=2, type=int, default=None, help="Explicit stream pair to normalize, for example --normalize-stream-pair 0 1")
    parser.add_argument("--normalize-use-preprocess-recommendation", action="store_true", help="Use preprocess pairwise recommendations when selecting the multistream pair.")
    parser.add_argument("--normalize-output-format", type=str, default="auto", help="Normalized output format written into the normalization metadata, for example auto, dual-fisheye, dfisheye, or flat.")
    parser.add_argument("--normalize-layout", choices=["auto", "side_by_side_lr", "side_by_side_rl", "top_bottom_tb", "top_bottom_bt"], default="auto")
    parser.add_argument("--normalize-resize-streams-to", choices=["match-first", "max", "none"], default="match-first")
    parser.add_argument("--normalize-rotate-a", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--normalize-rotate-b", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--normalize-flip-h-a", action="store_true")
    parser.add_argument("--normalize-flip-v-a", action="store_true")
    parser.add_argument("--normalize-flip-h-b", action="store_true")
    parser.add_argument("--normalize-flip-v-b", action="store_true")
    parser.add_argument("--normalize-output-prefix", type=str, default=None, help="Output prefix for normalized flat frames. Defaults to --frame-prefix.")
    parser.add_argument("--normalize-limit", type=int, default=None)
    parser.add_argument("--clean-normalize", action="store_true")

    # Step 3
    parser.add_argument(
        "--input-format",
        type=str,
        default=None,
        help="Input projection format for FFmpeg v360, for example auto, equirect, fisheye, dual-fisheye, flat/plain, eac, cubemap.",
    )
    parser.add_argument("--input-h-fov", type=float, default=None)
    parser.add_argument("--input-v-fov", type=float, default=None)
    parser.add_argument("--input-d-fov", type=float, default=None)
    parser.add_argument("--views", nargs="+", default=None)
    parser.add_argument("--h-fov", type=float, default=None)
    parser.add_argument("--v-fov", type=float, default=None)
    parser.add_argument("--d-fov", type=float, default=None)
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

    if args.preprocess_sample_count < 1:
        parser.error("--preprocess-sample-count must be at least 1.")

    if args.extract_video_stream_index is not None and args.extract_all_real_video_streams:
        parser.error("Use either --extract-video-stream-index or --extract-all-real-video-streams, not both.")

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
        logger.info("Preset: %s", args.preset)

        selected_steps = steps_to_run(args.step_from, args.step_to)
        logger.info("Steps to run: %s", selected_steps)

        if (args.extract_all_real_video_streams or args.extract_use_preprocess_recommendation) and "convert_360_to_views" in selected_steps:
            logger.warning(
                "Multistream extraction can populate data/frames_360/streams/stream_XX for later normalization or stitching, but convert_360_to_views still expects flat single-stream frames unless you add a normalizer step next."
            )

        python_exe = sys.executable
        logger.info("Python executable: %s", python_exe)

        commands: list[tuple[str, list[str]]] = []

        if "preprocess_input_video" in selected_steps:
            if args.preprocess_mode == "off":
                logger.info("Skipping preprocess_input_video because --preprocess-mode is off")
            else:
                commands.append(("preprocess_input_video", build_preprocess_cmd(paths, args, python_exe)))
        if "extract_frames" in selected_steps:
            commands.append(("extract_frames", build_extract_frames_cmd(paths, args, python_exe)))
        if "normalize_multistream_360" in selected_steps and args.normalize_multistream_mode != "off":
            commands.append(("normalize_multistream_360", build_normalize_multistream_cmd(paths, args, python_exe)))
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
            result = run_command_streaming(
                cmd,
                logger,
                verbose=args.verbose,
                cwd=code_root,
                step_name=step_name,
            )
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
