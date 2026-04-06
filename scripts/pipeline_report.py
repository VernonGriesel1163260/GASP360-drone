from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ./scripts importable when running:
# python .\scripts\pipeline_report.py
sys.path.append(str(Path(__file__).resolve().parent))

from common.logging_utils import setup_logger


SCRIPT_NAME = "pipeline_report"


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_dirs(root: Path, logger) -> dict[str, Path]:
    paths = {
        "root": root,
        "data": root / "data",
        "frames_360": root / "data" / "frames_360",
        "frames_perspective": root / "data" / "frames_perspective",
        "colmap": root / "data" / "colmap",
        "colmap_images": root / "data" / "colmap" / "images",
        "colmap_sparse": root / "data" / "colmap" / "sparse",
        "logs": root / "logs",
        "scripts": root / "scripts",
    }

    for key in ("data", "frames_360", "frames_perspective", "colmap", "colmap_images", "colmap_sparse", "logs", "scripts"):
        paths[key].mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s -> %s", key, paths[key])

    return paths


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png"}
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def file_size_mb(path: Path) -> float:
    if not path.exists() or not path.is_file():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def summarize_frames_360(paths: dict[str, Path], logger) -> dict:
    count = count_images(paths["frames_360"])
    logger.info("360 frames count: %s", count)
    return {"count": count}


def summarize_perspective(paths: dict[str, Path], logger) -> dict:
    summary = {}
    total = 0

    if paths["frames_perspective"].exists():
        for child in sorted(paths["frames_perspective"].iterdir()):
            if child.is_dir():
                count = count_images(child)
                summary[child.name] = count
                total += count

    logger.info("Perspective total image count: %s", total)
    for view_name, count in summary.items():
        logger.info("Perspective view folder '%s': %s image(s)", view_name, count)

    return {"total": total, "by_view": summary}


def summarize_colmap_images(paths: dict[str, Path], logger) -> dict:
    count = count_images(paths["colmap_images"])
    logger.info("COLMAP flat image count: %s", count)
    return {"count": count}


def summarize_colmap_database(paths: dict[str, Path], logger) -> dict:
    db_path = paths["colmap"] / "database.db"
    exists = db_path.exists()
    size_mb = file_size_mb(db_path)
    logger.info("COLMAP database exists: %s", exists)
    logger.info("COLMAP database size (MB): %.2f", size_mb)
    return {"exists": exists, "size_mb": size_mb, "path": str(db_path)}


def summarize_sparse(paths: dict[str, Path], logger) -> dict:
    sparse_dir = paths["colmap_sparse"]
    model_dirs = [p for p in sparse_dir.iterdir() if p.is_dir()] if sparse_dir.exists() else []

    logger.info("Sparse model folder count: %s", len(model_dirs))
    models = []

    for model_dir in sorted(model_dirs):
        files_present = sorted(p.name for p in model_dir.iterdir() if p.is_file())
        logger.info("Sparse model folder: %s", model_dir)
        logger.info("  Files: %s", files_present)
        models.append(
            {
                "name": model_dir.name,
                "path": str(model_dir),
                "files_present": files_present,
            }
        )

    return {"count": len(model_dirs), "models": models}


def print_report(summary: dict) -> None:
    print("\n=== Pipeline Report ===")
    print(f"360 frames: {summary['frames_360']['count']}")
    print(f"Perspective images total: {summary['perspective']['total']}")
    for view_name, count in summary["perspective"]["by_view"].items():
        print(f"  - {view_name}: {count}")
    print(f"COLMAP images: {summary['colmap_images']['count']}")
    print(
        f"COLMAP database: exists={summary['database']['exists']} "
        f"size_mb={summary['database']['size_mb']:.2f}"
    )
    print(f"Sparse model folders: {summary['sparse']['count']}")
    for model in summary["sparse"]["models"]:
        print(f"  - {model['name']}: files={', '.join(model['files_present'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize pipeline outputs after a run."
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger, run_log_path, latest_log_path = setup_logger(SCRIPT_NAME, verbose=args.verbose)

    try:
        root = project_root_from_script()
        logger.info("Project root: %s", root)
        logger.info("Run log: %s", run_log_path)
        logger.info("Latest log: %s", latest_log_path)

        paths = ensure_dirs(root, logger)

        summary = {
            "frames_360": summarize_frames_360(paths, logger),
            "perspective": summarize_perspective(paths, logger),
            "colmap_images": summarize_colmap_images(paths, logger),
            "database": summarize_colmap_database(paths, logger),
            "sparse": summarize_sparse(paths, logger),
        }

        print_report(summary)
        logger.info("Pipeline report completed successfully")
        return 0

    except Exception as exc:
        logger.exception("Fatal report error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())