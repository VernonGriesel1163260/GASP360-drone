from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from common.logging_utils import setup_logger
from common.workspace import resolve_workspace_root

SCRIPT_NAME = "prepare_colmap_images"
VALID_EXTS = {".jpg", ".jpeg", ".png"}


def project_root_from_script() -> Path:
    return resolve_workspace_root(caller_file=__file__)


def ensure_dirs(root: Path, logger) -> dict[str, Path]:
    paths = {
        "data": root / "data",
        "frames_perspective": root / "data" / "frames_perspective",
        "colmap": root / "data" / "colmap",
        "colmap_images": root / "data" / "colmap" / "images",
        "logs": root / "logs",
        "scripts": root / "scripts",
    }

    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s -> %s", name, path)

    return paths


def detect_available_views(frames_perspective_dir: Path, logger) -> list[str]:
    views = sorted([p.name for p in frames_perspective_dir.iterdir() if p.is_dir()])
    logger.info("Auto-detected view folders: %s", views)
    return views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten perspective view images into data/colmap/images for COLMAP."
    )

    parser.add_argument(
        "--views",
        nargs="+",
        default=None,
        help="Views to include. If omitted, auto-detect subfolders under data/frames_perspective.",
    )
    parser.add_argument(
        "--input-prefix",
        type=str,
        default=None,
        help="Only include files whose base frame stem starts with this prefix, e.g. frame360",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default="copy")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def clear_directory_files(target_dir: Path, logger) -> int:
    removed = 0
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        for file_path in target_dir.glob(pattern):
            file_path.unlink()
            removed += 1
            logger.debug("Deleted old COLMAP image: %s", file_path)
    return removed


def get_base_stem(filename_stem: str, view_name: str) -> str | None:
    suffix = f"_{view_name}"
    if filename_stem.endswith(suffix):
        return filename_stem[: -len(suffix)]
    return None


def collect_view_files(
    frames_perspective_dir: Path,
    views: list[str],
    input_prefix: str | None,
    logger,
) -> dict[str, dict[str, Path]]:
    grouped: dict[str, dict[str, Path]] = defaultdict(dict)

    for view_name in views:
        view_dir = frames_perspective_dir / view_name
        if not view_dir.exists():
            logger.warning("Requested view directory does not exist: %s", view_dir)
            continue

        logger.debug("Scanning view directory: %s", view_dir)

        files = sorted(
            p for p in view_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTS
        )

        logger.debug("Found %s candidate files in %s", len(files), view_dir)

        for file_path in files:
            base_stem = get_base_stem(file_path.stem, view_name)
            if base_stem is None:
                logger.debug("Skipping file with unexpected naming pattern: %s", file_path.name)
                continue

            if input_prefix and not base_stem.startswith(input_prefix):
                continue

            grouped[base_stem][view_name] = file_path

    logger.debug("Collected grouped base frames: %s", len(grouped))
    return dict(sorted(grouped.items()))


def validate_groups(
    grouped: dict[str, dict[str, Path]],
    requested_views: list[str],
    strict: bool,
    logger,
) -> tuple[list[str], list[str]]:
    complete = []
    incomplete = []

    for base_stem, view_map in grouped.items():
        missing = [view for view in requested_views if view not in view_map]
        if missing:
            incomplete.append(base_stem)
            logger.warning("Base frame '%s' is missing views: %s", base_stem, missing)
        else:
            complete.append(base_stem)

    if strict and incomplete:
        raise RuntimeError(
            f"Strict mode enabled and {len(incomplete)} base frame(s) are incomplete."
        )

    logger.info("Complete base frames: %s", len(complete))
    logger.info("Incomplete base frames: %s", len(incomplete))
    return complete, incomplete


def copy_or_link_file(src: Path, dst: Path, mode: str, logger) -> None:
    if dst.exists():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
            logger.warning("Hardlink failed, fell back to copy: %s -> %s", src, dst)
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


def prepare_colmap_images(
    grouped: dict[str, dict[str, Path]],
    selected_base_stems: list[str],
    requested_views: list[str],
    output_dir: Path,
    copy_mode: str,
    logger,
) -> int:
    count = 0

    for base_stem in selected_base_stems:
        view_map = grouped[base_stem]

        for view_name in requested_views:
            if view_name not in view_map:
                continue

            src = view_map[view_name]
            dst = output_dir / src.name
            copy_or_link_file(src, dst, copy_mode, logger)
            count += 1

    return count


def main() -> int:
    args = parse_args()
    root = project_root_from_script()
    logger, run_log_path, latest_log_path = setup_logger(
        SCRIPT_NAME,
        verbose=args.verbose,
        workspace_root=root,
    )

    try:
        logger.info("Project root: %s", root)
        logger.info("Run log: %s", run_log_path)
        logger.info("Latest log: %s", latest_log_path)

        paths = ensure_dirs(root, logger)
        logger.info("Project folders checked and created if missing")

        requested_views = args.views or detect_available_views(paths["frames_perspective"], logger)
        if not requested_views:
            logger.error("No view folders found under %s", paths["frames_perspective"])
            return 1

        grouped = collect_view_files(
            frames_perspective_dir=paths["frames_perspective"],
            views=requested_views,
            input_prefix=args.input_prefix,
            logger=logger,
        )

        if not grouped:
            logger.error("No perspective images found for the requested filters")
            return 1

        logger.info("Found %s grouped base frame(s)", len(grouped))

        complete, incomplete = validate_groups(
            grouped=grouped,
            requested_views=requested_views,
            strict=args.strict,
            logger=logger,
        )

        selected_base_stems = complete if complete else list(grouped.keys())

        if args.limit is not None:
            selected_base_stems = selected_base_stems[: args.limit]
            logger.info("Applying limit: using first %s base frame(s)", len(selected_base_stems))

        if not selected_base_stems:
            logger.error("No base frames selected for COLMAP image preparation")
            return 1

        output_dir = paths["colmap_images"]
        logger.info("COLMAP image output directory: %s", output_dir)

        if args.clean:
            removed = clear_directory_files(output_dir, logger)
            logger.info("Removed %s existing file(s) from COLMAP image folder", removed)

        output_count = prepare_colmap_images(
            grouped=grouped,
            selected_base_stems=selected_base_stems,
            requested_views=requested_views,
            output_dir=output_dir,
            copy_mode=args.copy_mode,
            logger=logger,
        )

        logger.info("Prepared %s COLMAP image(s)", output_count)
        logger.info("Selected base frame count: %s", len(selected_base_stems))
        logger.info("Views included: %s", requested_views)
        logger.info("Preparation completed successfully")
        return 0

    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())