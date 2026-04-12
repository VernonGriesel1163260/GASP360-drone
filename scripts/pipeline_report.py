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
PROJECTION_METADATA_FILENAME = "_projection_metadata.json"
PREPROCESS_METADATA_FILENAME = "_preprocess_metadata.json"
EXTRACTION_METADATA_FILENAME = "_extraction_metadata.json"


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


def load_json_file(path: Path, logger) -> dict | None:
    if not path.exists() or not path.is_file():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse JSON file %s: %s", path, exc)
        return None


def summarize_preprocess_metadata(paths: dict[str, Path], logger) -> dict:
    metadata_path = paths["data"] / "input_video" / PREPROCESS_METADATA_FILENAME
    payload = load_json_file(metadata_path, logger)

    summary = {
        "exists": metadata_path.exists(),
        "path": str(metadata_path),
        "input_video_name": None,
        "mode": None,
        "candidate_stream_count": 0,
        "candidate_stream_indexes": [],
        "recommended_strategy": None,
        "recommended_primary_stream_index": None,
        "recommended_frame_format": None,
        "decision_source": None,
        "pairwise_classes": [],
        "raw": payload,
    }

    if not payload:
        logger.info("Preprocess metadata sidecar exists: %s", summary["exists"])
        return summary

    recommendation = payload.get("recommendation") or {}
    candidate_streams = payload.get("candidate_streams") or []
    pairwise_results = payload.get("pairwise_results") or []

    summary.update(
        {
            "input_video_name": payload.get("input_video_name"),
            "mode": payload.get("mode"),
            "candidate_stream_count": len(candidate_streams),
            "candidate_stream_indexes": [stream.get("stream_index") for stream in candidate_streams],
            "recommended_strategy": recommendation.get("recommended_strategy"),
            "recommended_primary_stream_index": recommendation.get("recommended_primary_stream_index"),
            "recommended_frame_format": recommendation.get("recommended_frame_format"),
            "decision_source": recommendation.get("decision_source"),
            "pairwise_classes": [pair.get("classification") for pair in pairwise_results],
        }
    )

    logger.info("Preprocess metadata sidecar exists: %s", summary["exists"])
    logger.info("Preprocess metadata input video: %s", summary["input_video_name"])
    logger.info("Preprocess metadata mode: %s", summary["mode"])
    logger.info("Preprocess metadata candidate streams: %s", summary["candidate_stream_indexes"])
    logger.info("Preprocess metadata recommended strategy: %s", summary["recommended_strategy"])
    logger.info("Preprocess metadata recommended primary stream index: %s", summary["recommended_primary_stream_index"])
    logger.info("Preprocess metadata recommended frame format: %s", summary["recommended_frame_format"])
    logger.info("Preprocess metadata decision source: %s", summary["decision_source"])

    return summary


def summarize_extraction_metadata(paths: dict[str, Path], logger) -> dict:
    metadata_path = paths["frames_360"] / EXTRACTION_METADATA_FILENAME
    payload = load_json_file(metadata_path, logger)

    summary = {
        "exists": metadata_path.exists(),
        "path": str(metadata_path),
        "output_layout": None,
        "extraction_mode": None,
        "selected_stream_indexes": [],
        "stream_frame_counts": {},
        "flat_frame_count": 0,
        "total_frame_count": 0,
        "raw": payload,
    }

    if not payload:
        logger.info("Extraction metadata sidecar exists: %s", summary["exists"])
        return summary

    selected_streams = payload.get("selected_streams") or []
    stream_frame_counts = {f"stream_{item.get('stream_index'):02d}": item.get("frame_count") for item in selected_streams if item.get("stream_index") is not None}
    total_frame_count = sum(int(item.get("frame_count") or 0) for item in selected_streams)
    if payload.get("output_layout") == "flat" and len(selected_streams) == 1:
        flat_frame_count = int(selected_streams[0].get("frame_count") or 0)
    else:
        flat_frame_count = 0

    summary.update(
        {
            "output_layout": payload.get("output_layout"),
            "extraction_mode": payload.get("extraction_mode"),
            "selected_stream_indexes": [item.get("stream_index") for item in selected_streams],
            "stream_frame_counts": stream_frame_counts,
            "flat_frame_count": flat_frame_count,
            "total_frame_count": total_frame_count,
        }
    )

    logger.info("Extraction metadata sidecar exists: %s", summary["exists"])
    logger.info("Extraction metadata output layout: %s", summary["output_layout"])
    logger.info("Extraction metadata extraction mode: %s", summary["extraction_mode"])
    logger.info("Extraction metadata selected stream indexes: %s", summary["selected_stream_indexes"])
    logger.info("Extraction metadata stream frame counts: %s", summary["stream_frame_counts"])

    return summary


def summarize_normalization_metadata(paths: dict[str, Path], logger) -> dict:
    metadata_path = paths["frames_360"] / NORMALIZATION_METADATA_FILENAME
    payload = load_json_file(metadata_path, logger)

    summary = {
        "exists": metadata_path.exists(),
        "path": str(metadata_path),
        "mode": None,
        "pair_selection_mode": None,
        "selected_pair": [],
        "resolved_output_format": None,
        "resolved_layout": None,
        "effective_convert_input_format": None,
        "output_prefix": None,
        "output_frame_count": 0,
        "raw": payload,
    }

    if not payload:
        logger.info("Normalization metadata sidecar exists: %s", summary["exists"])
        return summary

    selected_pair = payload.get("selected_pair") or []
    summary.update(
        {
            "mode": payload.get("mode"),
            "pair_selection_mode": payload.get("pair_selection_mode"),
            "selected_pair": [item.get("stream_index") for item in selected_pair],
            "resolved_output_format": payload.get("resolved_output_format"),
            "resolved_layout": payload.get("resolved_layout"),
            "effective_convert_input_format": payload.get("effective_convert_input_format"),
            "output_prefix": payload.get("output_prefix"),
            "output_frame_count": int(payload.get("output_frame_count") or 0),
        }
    )

    logger.info("Normalization metadata sidecar exists: %s", summary["exists"])
    logger.info("Normalization metadata mode: %s", summary["mode"])
    logger.info("Normalization metadata pair selection mode: %s", summary["pair_selection_mode"])
    logger.info("Normalization metadata selected pair: %s", summary["selected_pair"])
    logger.info("Normalization metadata resolved output format: %s", summary["resolved_output_format"])
    logger.info("Normalization metadata resolved layout: %s", summary["resolved_layout"])
    logger.info("Normalization metadata effective convert input format: %s", summary["effective_convert_input_format"])
    logger.info("Normalization metadata output prefix: %s", summary["output_prefix"])
    logger.info("Normalization metadata output frame count: %s", summary["output_frame_count"])

    return summary


def summarize_projection_metadata(paths: dict[str, Path], logger) -> dict:
    metadata_path = paths["frames_360"] / PROJECTION_METADATA_FILENAME
    payload = load_json_file(metadata_path, logger)

    summary = {
        "exists": metadata_path.exists(),
        "path": str(metadata_path),
        "requested_input_format": None,
        "resolved_input_format": None,
        "detection_mode": None,
        "input_frame_count": None,
        "sample_frame_name": None,
        "confidence": None,
        "output_views": [],
        "output_view_count": None,
        "output_width": None,
        "output_height": None,
        "metrics": None,
        "raw": payload,
    }

    if not payload:
        logger.info("Projection metadata sidecar exists: %s", summary["exists"])
        return summary

    auto_detection = payload.get("auto_detection") or {}
    output_projection = payload.get("output_projection") or {}

    summary.update(
        {
            "requested_input_format": payload.get("requested_input_format"),
            "resolved_input_format": payload.get("resolved_input_format"),
            "detection_mode": payload.get("detection_mode"),
            "input_frame_count": payload.get("input_frame_count"),
            "sample_frame_name": auto_detection.get("sample_frame_name"),
            "confidence": auto_detection.get("confidence"),
            "output_views": output_projection.get("views") or [],
            "output_view_count": len(output_projection.get("views") or []),
            "output_width": output_projection.get("width"),
            "output_height": output_projection.get("height"),
            "metrics": auto_detection.get("metrics"),
        }
    )

    logger.info("Projection metadata sidecar exists: %s", summary["exists"])
    logger.info("Projection metadata requested input format: %s", summary["requested_input_format"])
    logger.info("Projection metadata resolved input format: %s", summary["resolved_input_format"])
    logger.info("Projection metadata detection mode: %s", summary["detection_mode"])
    logger.info("Projection metadata input frame count: %s", summary["input_frame_count"])
    logger.info("Projection metadata sample frame: %s", summary["sample_frame_name"])
    logger.info("Projection metadata confidence: %s", summary["confidence"])
    logger.info("Projection metadata output views: %s", summary["output_views"])
    logger.info("Projection metadata output size: %sx%s", summary["output_width"], summary["output_height"])

    return summary


def summarize_frames_360(paths: dict[str, Path], logger) -> dict:
    count = count_images(paths["frames_360"])
    preview = []
    if paths["frames_360"].exists():
        preview = [p.name for p in sorted(paths["frames_360"].iterdir()) if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}][:10]

    streams_root = paths["frames_360"] / "streams"
    by_stream = {}
    stream_total = 0
    if streams_root.exists():
        for child in sorted(streams_root.iterdir()):
            if child.is_dir():
                child_count = count_images(child)
                by_stream[child.name] = child_count
                stream_total += child_count

    logger.info("360 flat frames count: %s", count)
    logger.info("360 multistream total frame count: %s", stream_total)
    for stream_name, child_count in by_stream.items():
        logger.info("360 multistream folder '%s': %s image(s)", stream_name, child_count)

    return {"count": count, "preview": preview, "streams_root": str(streams_root), "by_stream": by_stream, "stream_total": stream_total}


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
    preprocess = summary["preprocess"]
    print("Preprocess summary:")
    print(f"  sidecar_exists={preprocess['exists']}")
    print(f"  input_video_name={preprocess['input_video_name']}")
    print(f"  mode={preprocess['mode']}")
    print(f"  candidate_stream_count={preprocess['candidate_stream_count']}")
    print(f"  candidate_stream_indexes={preprocess['candidate_stream_indexes']}")
    print(f"  recommended_strategy={preprocess['recommended_strategy']}")
    print(f"  recommended_primary_stream_index={preprocess['recommended_primary_stream_index']}")
    print(f"  recommended_frame_format={preprocess['recommended_frame_format']}")
    extraction = summary["extraction"]
    print("Extraction summary:")
    print(f"  sidecar_exists={extraction['exists']}")
    print(f"  output_layout={extraction['output_layout']}")
    print(f"  extraction_mode={extraction['extraction_mode']}")
    print(f"  selected_stream_indexes={extraction['selected_stream_indexes']}")
    print(f"  stream_frame_counts={extraction['stream_frame_counts']}")
    normalization = summary["normalization"]
    print("Normalization summary:")
    print(f"  sidecar_exists={normalization['exists']}")
    print(f"  mode={normalization['mode']}")
    print(f"  pair_selection_mode={normalization['pair_selection_mode']}")
    print(f"  selected_pair={normalization['selected_pair']}")
    print(f"  resolved_output_format={normalization['resolved_output_format']}")
    print(f"  resolved_layout={normalization['resolved_layout']}")
    print(f"  effective_convert_input_format={normalization['effective_convert_input_format']}")
    print(f"  output_prefix={normalization['output_prefix']}")
    print(f"  output_frame_count={normalization['output_frame_count']}")
    print(f"360 flat frames: {summary['frames_360']['count']}")
    print(f"360 multistream total: {summary['frames_360']['stream_total']}")
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
        "preprocess": summary.get("preprocess"),
        "extraction": summary.get("extraction"),
        "projection": summary.get("projection_metadata"),
        "normalization": summary.get("normalization"),
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
        "extraction": {
            "metadata_sidecar": summary["extraction"].get("path"),
            "output_layout": summary["extraction"].get("output_layout"),
            "extraction_mode": summary["extraction"].get("extraction_mode"),
            "selected_stream_indexes": summary["extraction"].get("selected_stream_indexes"),
            "stream_frame_counts": summary["extraction"].get("stream_frame_counts"),
            "flat_frame_count": summary["extraction"].get("flat_frame_count"),
            "total_frame_count": summary["extraction"].get("total_frame_count"),
        },
        "normalization": {
            "metadata_sidecar": summary["normalization"].get("path"),
            "mode": summary["normalization"].get("mode"),
            "pair_selection_mode": summary["normalization"].get("pair_selection_mode"),
            "selected_pair": summary["normalization"].get("selected_pair"),
            "resolved_output_format": summary["normalization"].get("resolved_output_format"),
            "resolved_layout": summary["normalization"].get("resolved_layout"),
            "effective_convert_input_format": summary["normalization"].get("effective_convert_input_format"),
            "output_prefix": summary["normalization"].get("output_prefix"),
            "output_frame_count": summary["normalization"].get("output_frame_count"),
        },
        "projection": {
            "metadata_sidecar": summary["projection_metadata"].get("path"),
            "requested_input_format": summary["projection_metadata"].get("requested_input_format"),
            "resolved_input_format": summary["projection_metadata"].get("resolved_input_format"),
            "detection_mode": summary["projection_metadata"].get("detection_mode"),
            "sample_frame_name": summary["projection_metadata"].get("sample_frame_name"),
            "confidence": summary["projection_metadata"].get("confidence"),
            "output_views": summary["projection_metadata"].get("output_views"),
            "output_width": summary["projection_metadata"].get("output_width"),
            "output_height": summary["projection_metadata"].get("output_height"),
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
            "preprocess": summarize_preprocess_metadata(paths, logger),
            "frames_360": summarize_frames_360(paths, logger),
            "extraction": summarize_extraction_metadata(paths, logger),
            "normalization": summarize_normalization_metadata(paths, logger),
            "projection_metadata": summarize_projection_metadata(paths, logger),
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
