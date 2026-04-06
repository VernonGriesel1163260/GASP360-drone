from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

# Make ./scripts importable when running:
# python .\scripts\inspect_colmap_models.py
sys.path.append(str(Path(__file__).resolve().parent))

from common.logging_utils import setup_logger


SCRIPT_NAME = "inspect_colmap_models"


@dataclass
class ModelStats:
    name: str
    path: str
    registered_images: int = 0
    total_input_images: int = 0
    registration_ratio: float = 0.0
    points3D: int = 0
    observations: int = 0
    mean_track_length: float = 0.0
    mean_observations_per_image: float = 0.0
    reprojection_error: float = 0.0
    analyzer_ok: bool = False
    raw_output: str = ""


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_dirs(root: Path, logger) -> dict[str, Path]:
    paths = {
        "root": root,
        "data": root / "data",
        "colmap": root / "data" / "colmap",
        "colmap_sparse": root / "data" / "colmap" / "sparse",
        "colmap_images": root / "data" / "colmap" / "images",
        "colmap_best": root / "data" / "colmap" / "sparse_best",
        "logs": root / "logs",
        "scripts": root / "scripts",
        "colmap_exe": root / "COLMAP" / "bin" / "colmap.exe",
        "colmap_bat": root / "COLMAP" / "COLMAP.bat",
    }

    for key in ("data", "colmap", "colmap_sparse", "colmap_images", "colmap_best", "logs", "scripts"):
        paths[key].mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s -> %s", key, paths[key])

    return paths


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png"}
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def quote_cmd(cmd: Iterable[str]) -> str:
    return " ".join(f'"{c}"' if " " in c else c for c in cmd)


def stream_process_output(process: subprocess.Popen, logger, prefix: str, verbose: bool) -> str:
    assert process.stdout is not None
    lines: list[str] = []

    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if not line:
            continue

        lines.append(line)
        logger.debug("%s %s", prefix, line)
        if verbose:
            print(f"{prefix} {line}")

    return "\n".join(lines)


def run_command_capture(cmd: list[str], logger, verbose: bool = False, cwd: Path | None = None) -> tuple[int, str]:
    logger.info("Running command")
    logger.debug("Command: %s", quote_cmd(cmd))
    if cwd is not None:
        logger.debug("Working directory: %s", cwd)

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
        output = stream_process_output(process, logger, prefix="[SUBPROCESS]", verbose=verbose)
    finally:
        return_code = process.wait()

    logger.debug("Command exit code: %s", return_code)
    return return_code, output


def detect_colmap_exe(paths: dict[str, Path], logger) -> Path | None:
    exe_path = paths["colmap_exe"]
    if exe_path.exists():
        logger.info("Detected colmap.exe: %s", exe_path)
        return exe_path

    logger.info("colmap.exe not found at expected path: %s", exe_path)
    return None


def detect_colmap_bat(paths: dict[str, Path], logger) -> Path | None:
    bat_path = paths["colmap_bat"]
    if bat_path.exists():
        logger.info("Detected COLMAP.bat: %s", bat_path)
        return bat_path

    logger.info("COLMAP.bat not found at expected path: %s", bat_path)
    return None


def find_model_dirs(sparse_dir: Path) -> list[Path]:
    if not sparse_dir.exists():
        return []
    return sorted([p for p in sparse_dir.iterdir() if p.is_dir()], key=lambda p: p.name)


def build_model_analyzer_cmd(colmap_cmd: Path, model_path: Path) -> list[str]:
    return [
        str(colmap_cmd),
        "model_analyzer",
        "--path",
        str(model_path),
    ]


def parse_int(pattern: str, text: str) -> int:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return 0
    return int(m.group(1))


def parse_float(pattern: str, text: str) -> float:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return 0.0
    return float(m.group(1))


def parse_model_analyzer_output(model_dir: Path, output: str, total_input_images: int) -> ModelStats:
    stats = ModelStats(
        name=model_dir.name,
        path=str(model_dir),
        analyzer_ok=True,
        raw_output=output,
        total_input_images=total_input_images,
    )

    stats.registered_images = parse_int(r"Registered images\s*:\s*(\d+)", output)
    stats.points3D = parse_int(r"Points(?:3D)?\s*:\s*(\d+)", output)
    stats.observations = parse_int(r"Observations\s*:\s*(\d+)", output)
    stats.mean_track_length = parse_float(r"Mean track length\s*:\s*([0-9.]+)", output)
    stats.mean_observations_per_image = parse_float(r"Mean observations per image\s*:\s*([0-9.]+)", output)
    stats.reprojection_error = parse_float(r"Mean reprojection error\s*:\s*([0-9.]+)", output)

    if total_input_images > 0:
        stats.registration_ratio = stats.registered_images / total_input_images

    return stats


def fallback_binary_presence_stats(model_dir: Path, total_input_images: int) -> ModelStats:
    return ModelStats(
        name=model_dir.name,
        path=str(model_dir),
        analyzer_ok=False,
        raw_output="",
        total_input_images=total_input_images,
        registration_ratio=0.0,
    )


def inspect_model(colmap_cmd: Path, model_dir: Path, total_input_images: int, logger, verbose: bool) -> ModelStats:
    cmd = build_model_analyzer_cmd(colmap_cmd, model_dir)
    exit_code, output = run_command_capture(cmd, logger, verbose=verbose)

    if exit_code != 0:
        logger.warning("model_analyzer failed for %s", model_dir)
        return fallback_binary_presence_stats(model_dir, total_input_images)

    return parse_model_analyzer_output(model_dir, output, total_input_images)


def sort_models(models: list[ModelStats]) -> list[ModelStats]:
    return sorted(
        models,
        key=lambda m: (
            m.registered_images,
            m.points3D,
            m.observations,
            m.mean_observations_per_image,
            -int(m.name) if m.name.isdigit() else 0,
        ),
        reverse=True,
    )


def promote_best_model(best_model: ModelStats, target_dir: Path, logger, mode: str) -> None:
    src_dir = Path(best_model.path)

    if target_dir.exists():
        shutil.rmtree(target_dir)
        logger.debug("Removed existing best-model directory: %s", target_dir)

    if mode == "copy":
        shutil.copytree(src_dir, target_dir)
        logger.info("Copied best model to: %s", target_dir)
    elif mode == "junction":
        shutil.copytree(src_dir, target_dir)
        logger.warning("Junction mode requested; copied instead for safety: %s", target_dir)
    else:
        raise ValueError(f"Unsupported promote mode: {mode}")


def write_summary_json(models: list[ModelStats], best_model: ModelStats | None, output_path: Path, logger) -> None:
    payload = {
        "best_model": asdict(best_model) if best_model else None,
        "models": [asdict(m) for m in models],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote JSON summary: %s", output_path)


def write_best_model_text(best_model: ModelStats | None, output_path: Path, logger) -> None:
    if best_model is None:
        output_path.write_text("No best model selected.\n", encoding="utf-8")
    else:
        output_path.write_text(
            f"{best_model.name}\n"
            f"{best_model.path}\n"
            f"registered_images={best_model.registered_images}\n"
            f"total_input_images={best_model.total_input_images}\n"
            f"registration_ratio={best_model.registration_ratio:.4f}\n",
            encoding="utf-8",
        )
    logger.info("Wrote best-model text file: %s", output_path)


def print_summary(models: list[ModelStats], best_model: ModelStats | None, total_input_images: int) -> None:
    print("\n=== COLMAP Model Inspection ===")
    print(f"Model count: {len(models)}")
    print(f"Total COLMAP input images: {total_input_images}")
    for m in models:
        print(
            f"- {m.name}: registered_images={m.registered_images}, "
            f"registration_ratio={m.registration_ratio:.2%}, "
            f"points3D={m.points3D}, observations={m.observations}, "
            f"mean_track_length={m.mean_track_length:.2f}, "
            f"mean_obs_per_image={m.mean_observations_per_image:.2f}, "
            f"reproj_error={m.reprojection_error:.4f}, analyzer_ok={m.analyzer_ok}"
        )
    if best_model:
        print("\nBest model:")
        print(
            f"  {best_model.name} "
            f"(registered_images={best_model.registered_images}, "
            f"registration_ratio={best_model.registration_ratio:.2%}, "
            f"points3D={best_model.points3D}, observations={best_model.observations})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect COLMAP sparse models, count registered images, and choose the best one automatically."
    )

    parser.add_argument("--sparse-dir", type=str, default=None)
    parser.add_argument("--promote-best", action="store_true")
    parser.add_argument("--promote-mode", choices=["copy", "junction"], default="copy")
    parser.add_argument("--summary-json", action="store_true")
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

        sparse_dir = Path(args.sparse_dir).resolve() if args.sparse_dir else paths["colmap_sparse"]
        colmap_images_dir = paths["colmap_images"]
        total_input_images = count_images(colmap_images_dir)

        logger.info("Sparse root: %s", sparse_dir)
        logger.info("Total COLMAP input images: %s", total_input_images)

        if not sparse_dir.exists():
            logger.error("Sparse directory does not exist: %s", sparse_dir)
            return 1

        colmap_exe = detect_colmap_exe(paths, logger)
        colmap_bat = detect_colmap_bat(paths, logger)

        colmap_cmd = colmap_exe or colmap_bat
        if colmap_cmd is None:
            logger.error("No COLMAP command available for model inspection.")
            return 1

        model_dirs = find_model_dirs(sparse_dir)
        if not model_dirs:
            logger.error("No sparse model directories found in: %s", sparse_dir)
            return 1

        logger.info("Found %s sparse model folder(s)", len(model_dirs))

        models: list[ModelStats] = []
        for model_dir in model_dirs:
            logger.info("Inspecting model: %s", model_dir)
            stats = inspect_model(colmap_cmd, model_dir, total_input_images, logger, verbose=args.verbose)
            models.append(stats)

        ranked = sort_models(models)
        best_model = ranked[0] if ranked else None

        for idx, model in enumerate(ranked, start=1):
            logger.info(
                "Rank %s | %s | registered_images=%s | registration_ratio=%.2f%% | points3D=%s | observations=%s",
                idx,
                model.name,
                model.registered_images,
                model.registration_ratio * 100.0,
                model.points3D,
                model.observations,
            )

        if best_model:
            logger.info("Best model selected: %s", best_model.name)

        best_txt = paths["colmap"] / "best_model.txt"
        write_best_model_text(best_model, best_txt, logger)

        if args.summary_json:
            json_path = paths["colmap"] / "model_inspection.json"
            write_summary_json(ranked, best_model, json_path, logger)

        if args.promote_best and best_model is not None:
            promote_best_model(best_model, paths["colmap_best"], logger, args.promote_mode)

        print_summary(ranked, best_model, total_input_images)
        logger.info("Model inspection completed successfully")
        return 0

    except Exception as exc:
        logger.exception("Fatal inspection error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())