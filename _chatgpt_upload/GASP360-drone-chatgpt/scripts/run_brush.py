from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

# Make ./scripts importable when running:
sys.path.append(str(Path(__file__).resolve().parent))

from common.logging_utils import setup_logger
from common.workspace import resolve_code_root, resolve_workspace_root


SCRIPT_NAME = "brush_reconstruction"


def code_root_from_script() -> Path:
    return resolve_code_root(__file__)


def workspace_root_from_script() -> Path:
    return resolve_workspace_root(caller_file=__file__)


def ensure_dirs(code_root: Path, workspace_root: Path, logger) -> dict[str, Path]:
    paths = {
        "code_root": code_root,
        "workspace_root": workspace_root,
        "data": workspace_root / "data",
        "colmap": workspace_root / "data" / "colmap",
        "colmap_images": workspace_root / "data" / "colmap" / "images",
        "colmap_sparse": workspace_root / "data" / "colmap" / "sparse",
        "colmap_sparse_best": workspace_root / "data" / "colmap" / "sparse_best",
        "brush": workspace_root / "data" / "brush",
        "brush_input_colmap": workspace_root / "data" / "brush" / "input_colmap",
        "brush_input_images": workspace_root / "data" / "brush" / "input_colmap" / "images",
        "brush_input_sparse": workspace_root / "data" / "brush" / "input_colmap" / "sparse" / "0",
        "brush_output": workspace_root / "data" / "brush" / "output",
        "logs": workspace_root / "logs",
        "workspace_context_root": workspace_root / "workspace_context.json",
        "workspace_context_data": workspace_root / "data" / "workspace_context.json",
        "brush_exe_default": code_root / "tools" / "brush" / "brush_app.exe",
        "brush_exe_alt": code_root / "tools" / "brush" / "brush.exe",
        "brush_latest_path": code_root / "tools" / "brush" / "brush_latest_path.txt",
    }

    for key in (
        "data",
        "colmap",
        "colmap_images",
        "colmap_sparse",
        "brush",
        "brush_input_colmap",
        "brush_input_images",
        "brush_input_sparse",
        "brush_output",
        "logs",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s -> %s", key, paths[key])

    return paths


def quote_cmd(cmd: Iterable[str]) -> str:
    return " ".join(f'"{c}"' if " " in c else c for c in cmd)


def read_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_workspace_context(paths: dict[str, Path], explicit_context: str | None, logger) -> dict | None:
    candidates: list[Path] = []
    if explicit_context:
        candidates.append(Path(explicit_context).expanduser().resolve())
    candidates.extend([
        paths["workspace_context_root"],
        paths["workspace_context_data"],
    ])

    for candidate in candidates:
        payload = read_json_if_exists(candidate)
        if payload is not None:
            logger.info("Using workspace context: %s", candidate)
            return payload

    logger.info("No workspace_context.json found")
    return None


def detect_brush_exe(paths: dict[str, Path], override: str | None, logger) -> Path:
    candidates: list[Path] = []

    if override:
        candidates.append(Path(override).expanduser().resolve())

    latest_path_file = paths["brush_latest_path"]
    if latest_path_file.exists():
        try:
            latest_text = latest_path_file.read_text(encoding="utf-8").strip()
            if latest_text:
                candidates.append(Path(latest_text).expanduser().resolve())
        except Exception:
            pass

    candidates.extend([
        paths["brush_exe_default"],
        paths["brush_exe_alt"],
    ])

    for candidate in candidates:
        if candidate.exists():
            logger.info("Detected Brush executable: %s", candidate)
            return candidate

    raise FileNotFoundError(
        "Brush executable not found. Checked:\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )


def clear_directory(path: Path, logger) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
            logger.debug("Deleted file: %s", child)
        elif child.is_dir():
            shutil.rmtree(child)
            logger.debug("Deleted directory: %s", child)


def copy_or_link_file(src: Path, dst: Path, mode: str, logger) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        logger.debug("Copied: %s -> %s", src, dst)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
            logger.debug("Hardlinked: %s -> %s", src, dst)
        except OSError:
            shutil.copy2(src, dst)
            logger.warning("Hardlink failed, fell back to copy: %s -> %s", src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def resolve_best_sparse_model(paths: dict[str, Path], context: dict | None, logger) -> Path | None:
    if context:
        colmap = context.get("colmap", {})
        for key in ("sparse_best", "best_model_path"):
            value = colmap.get(key)
            if value:
                candidate = Path(value).expanduser().resolve()
                if candidate.exists():
                    logger.info("Resolved sparse model from context (%s): %s", key, candidate)
                    return candidate

    best_dir = paths["colmap_sparse_best"]
    if best_dir.exists() and any(best_dir.iterdir()):
        logger.info("Resolved sparse model from sparse_best: %s", best_dir)
        return best_dir

    if paths["colmap_sparse"].exists():
        subdirs = sorted([p for p in paths["colmap_sparse"].iterdir() if p.is_dir()], key=lambda p: p.name)
        if subdirs:
            logger.info("Resolved sparse model from first sparse dir: %s", subdirs[0])
            return subdirs[0]

    return None


def prepare_standard_colmap_dataset(
    images_dir: Path,
    sparse_model_dir: Path,
    brush_input_images: Path,
    brush_input_sparse: Path,
    mode: str,
    clean: bool,
    logger,
) -> Path:
    if clean:
        clear_directory(brush_input_images, logger)
        clear_directory(brush_input_sparse, logger)

    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts])

    if not image_files:
        raise RuntimeError(f"No image files found in {images_dir}")

    for src in image_files:
        dst = brush_input_images / src.name
        copy_or_link_file(src, dst, mode, logger)

    sparse_files = sorted([p for p in sparse_model_dir.iterdir() if p.is_file()])
    if not sparse_files:
        raise RuntimeError(f"No sparse model files found in {sparse_model_dir}")

    for src in sparse_files:
        dst = brush_input_sparse / src.name
        copy_or_link_file(src, dst, mode, logger)

    dataset_root = brush_input_images.parent.parent
    logger.info("Prepared Brush dataset root: %s", dataset_root)
    return dataset_root


def resolve_dataset_root(
    args,
    paths: dict[str, Path],
    context: dict | None,
    logger,
) -> Path:
    if args.dataset_root:
        dataset_root = Path(args.dataset_root).expanduser().resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
        logger.info("Using explicit dataset root: %s", dataset_root)
        return dataset_root

    images_dir: Path | None = None
    sparse_model_dir: Path | None = None

    if args.images_dir:
        images_dir = Path(args.images_dir).expanduser().resolve()
    elif context:
        colmap = context.get("colmap", {})
        if colmap.get("images"):
            images_dir = Path(colmap["images"]).expanduser().resolve()
    else:
        images_dir = paths["colmap_images"]

    if args.sparse_model:
        sparse_model_dir = Path(args.sparse_model).expanduser().resolve()
    else:
        sparse_model_dir = resolve_best_sparse_model(paths, context, logger)

    # Reuse already-prepared dataset if requested or available
    prepared_dataset = paths["brush_input_colmap"]
    if (
        not args.force_prepare
        and prepared_dataset.exists()
        and (prepared_dataset / "images").exists()
        and (prepared_dataset / "sparse" / "0").exists()
        and any((prepared_dataset / "images").iterdir())
    ):
        logger.info("Using existing prepared Brush dataset: %s", prepared_dataset)
        return prepared_dataset

    if images_dir is None or not images_dir.exists():
        raise FileNotFoundError(f"COLMAP images directory not found: {images_dir}")

    if sparse_model_dir is None or not sparse_model_dir.exists():
        raise FileNotFoundError(f"COLMAP sparse model directory not found: {sparse_model_dir}")

    logger.info("Preparing dataset from COLMAP images: %s", images_dir)
    logger.info("Preparing dataset from COLMAP sparse model: %s", sparse_model_dir)

    return prepare_standard_colmap_dataset(
        images_dir=images_dir,
        sparse_model_dir=sparse_model_dir,
        brush_input_images=paths["brush_input_images"],
        brush_input_sparse=paths["brush_input_sparse"],
        mode=args.copy_mode,
        clean=args.clean_input,
        logger=logger,
    )


def stream_process_output(process: subprocess.Popen, logger, prefix: str, verbose: bool) -> None:
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        logger.debug("%s %s", prefix, line)
        if verbose:
            print(f"{prefix} {line}")


def run_command_streaming(cmd: list[str], logger, verbose: bool = False, cwd: Path | None = None) -> None:
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
        stream_process_output(process, logger, "[SUBPROCESS]", verbose)
    finally:
        return_code = process.wait()

    logger.debug("Command exit code: %s", return_code)
    if return_code != 0:
        raise RuntimeError(f"Brush command failed with exit code {return_code}")


def write_brush_run_manifest(
    paths: dict[str, Path],
    brush_exe: Path,
    dataset_root: Path,
    cmd: list[str],
    logger,
) -> None:
    manifest = {
        "workspace_root": str(paths["workspace_root"]),
        "code_root": str(paths["code_root"]),
        "brush_exe": str(brush_exe),
        "dataset_root": str(dataset_root),
        "command": cmd,
    }
    manifest_path = paths["brush"] / "brush_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote Brush run manifest: %s", manifest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Brush with either an explicit dataset root or a prepared COLMAP dataset."
    )

    parser.add_argument("--brush-exe", type=str, default=None, help="Override Brush executable path")
    parser.add_argument("--dataset-root", type=str, default=None, help="Run Brush directly on this dataset root")
    parser.add_argument("--context-file", type=str, default=None, help="Optional workspace_context.json path")
    parser.add_argument("--images-dir", type=str, default=None, help="COLMAP images directory")
    parser.add_argument("--sparse-model", type=str, default=None, help="COLMAP sparse model directory")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare dataset but do not launch Brush")
    parser.add_argument("--force-prepare", action="store_true", help="Rebuild prepared dataset even if one exists")
    parser.add_argument("--clean-input", action="store_true", help="Clean prepared dataset directories first")
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default="hardlink")
    parser.add_argument("--with-viewer", action="store_true", help="Pass --with-viewer to Brush")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--brush-args",
        nargs=argparse.REMAINDER,
        help="Extra Brush CLI args passed after the dataset path. Use '--brush-args -- ...' if needed.",
    )

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

        context = find_workspace_context(paths, args.context_file, logger)
        brush_exe = detect_brush_exe(paths, args.brush_exe, logger)
        dataset_root = resolve_dataset_root(args, paths, context, logger)

        cmd = [str(brush_exe), str(dataset_root)]

        if args.with_viewer:
            cmd.append("--with-viewer")

        extra_args = args.brush_args or []
        if extra_args and extra_args[0] == "--":
            extra_args = extra_args[1:]
        cmd.extend(extra_args)

        write_brush_run_manifest(paths, brush_exe, dataset_root, cmd, logger)

        if args.prepare_only:
            logger.info("Prepare-only mode enabled; skipping Brush launch")
            return 0

        run_command_streaming(cmd, logger, verbose=args.verbose, cwd=code_root)
        logger.info("Brush run completed successfully")
        return 0

    except Exception as exc:
        logger.exception("Fatal Brush run error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())