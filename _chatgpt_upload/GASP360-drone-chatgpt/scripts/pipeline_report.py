from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ./scripts importable when running:
# python .\scripts\pipeline_report.py
sys.path.append(str(Path(__file__).resolve().parent))

from common.logging_utils import setup_logger
from common.workspace import resolve_workspace_root


SCRIPT_NAME = "pipeline_report"


def project_root_from_script() -> Path:
    return resolve_workspace_root(caller_file=__file__)


def ensure_dirs(root: Path, logger) -> dict[str, Path]:
    paths = {
        "root": root,
        "data": root / "data",
        "frames_360": root / "data" / "frames_360",
        "frames_perspective": root / "data" / "frames_perspective",
        "colmap": root / "data" / "colmap",
        "colmap_images": root / "data" / "colmap" / "images",
        "colmap_sparse": root / "data" / "colmap" / "sparse",
        "colmap_best": root / "data" / "colmap" / "sparse_best",
        "logs": root / "logs",
        "scripts": root / "scripts",
    }

    for key in ("data", "frames_360", "frames_perspective", "colmap", "colmap_images", "colmap_sparse", "colmap_best", "logs", "scripts"):
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


def summarize_best_model(paths: dict[str, Path], logger) -> dict:
    best_txt = paths["colmap"] / "best_model.txt"
    best_json = paths["colmap"] / "model_inspection.json"
    sparse_best = paths["colmap_best"]

    result = {
        "best_model_txt_exists": best_txt.exists(),
        "model_inspection_json_exists": best_json.exists(),
        "sparse_best_exists": sparse_best.exists(),
        "selected_model_name": None,
        "selected_model_path": None,
        "registered_images": None,
        "total_input_images": None,
        "registration_ratio": None,
    }

    if best_txt.exists():
        lines = [line.strip() for line in best_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) >= 2:
            result["selected_model_name"] = lines[0]
            result["selected_model_path"] = lines[1]
        for line in lines[2:]:
            if line.startswith("registered_images="):
                result["registered_images"] = int(line.split("=", 1)[1])
            elif line.startswith("total_input_images="):
                result["total_input_images"] = int(line.split("=", 1)[1])
            elif line.startswith("registration_ratio="):
                result["registration_ratio"] = float(line.split("=", 1)[1])

    elif best_json.exists():
        try:
            payload = json.loads(best_json.read_text(encoding="utf-8"))
            best_model = payload.get("best_model")
            if best_model:
                result["selected_model_name"] = best_model.get("name")
                result["selected_model_path"] = best_model.get("path")
                result["registered_images"] = best_model.get("registered_images")
                result["total_input_images"] = best_model.get("total_input_images")
                result["registration_ratio"] = best_model.get("registration_ratio")
        except Exception as exc:
            logger.warning("Failed to parse model inspection JSON: %s", exc)

    logger.info("Best model text exists: %s", result["best_model_txt_exists"])
    logger.info("Model inspection JSON exists: %s", result["model_inspection_json_exists"])
    logger.info("Sparse best exists: %s", result["sparse_best_exists"])
    logger.info("Selected best model name: %s", result["selected_model_name"])
    logger.info("Selected best model path: %s", result["selected_model_path"])
    logger.info("Best model registered images: %s", result["registered_images"])
    logger.info("Best model total input images: %s", result["total_input_images"])
    logger.info("Best model registration ratio: %s", result["registration_ratio"])

    return result


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

    best = summary["best_model"]
    print("\nBest model summary:")
    print(f"  best_model_txt_exists={best['best_model_txt_exists']}")
    print(f"  model_inspection_json_exists={best['model_inspection_json_exists']}")
    print(f"  sparse_best_exists={best['sparse_best_exists']}")
    print(f"  selected_model_name={best['selected_model_name']}")
    print(f"  selected_model_path={best['selected_model_path']}")
    print(f"  registered_images={best['registered_images']}")
    print(f"  total_input_images={best['total_input_images']}")
    ratio = best["registration_ratio"]
    ratio_display = f"{ratio:.2%}" if isinstance(ratio, float) else str(ratio)
    print(f"  registration_ratio={ratio_display}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize pipeline outputs after a run."
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def write_workspace_context(paths: dict[str, Path], summary: dict, logger) -> None:
    best = summary["best_model"]

    payload = {
        "workspace_root": str(paths["root"]),
        "data_root": str(paths["data"]),
        "dataset_type": "colmap",
        "colmap": {
            "root": str(paths["colmap"]),
            "images": str(paths["colmap_images"]),
            "database": str(paths["colmap"] / "database.db"),
            "sparse_root": str(paths["colmap_sparse"]),
            "sparse_best": str(paths["colmap_best"]),
            "best_model_name": best["selected_model_name"],
            "best_model_path": best["selected_model_path"],
            "best_model_found": best["best_model_txt_exists"] or best["model_inspection_json_exists"],
        },
        "metrics": {
            "registered_images": best["registered_images"],
            "total_input_images": best["total_input_images"],
            "registration_ratio": best["registration_ratio"],
        },
        "brush": {
            "dataset_root_hint": str(paths["colmap"]),
            "prepared_input_root": str(paths["data"] / "brush" / "input_colmap"),
            "output_root": str(paths["data"] / "brush"),
        },
    }

    context_paths = [
        paths["root"] / "workspace_context.json",
        paths["data"] / "workspace_context.json",
    ]

    for context_path in context_paths:
        context_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote workspace context: %s", context_path)


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

        summary = {
            "frames_360": summarize_frames_360(paths, logger),
            "perspective": summarize_perspective(paths, logger),
            "colmap_images": summarize_colmap_images(paths, logger),
            "database": summarize_colmap_database(paths, logger),
            "sparse": summarize_sparse(paths, logger),
            "best_model": summarize_best_model(paths, logger),
        }

        write_workspace_context(paths, summary, logger)
        print_report(summary)
        logger.info("Pipeline report completed successfully")
        return 0

    except Exception as exc:
        logger.exception("Fatal report error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
