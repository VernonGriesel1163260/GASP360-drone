from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

sys.path.append(str(Path(__file__).resolve().parent))

from common.logging_utils import setup_logger
from common.presets import DEFAULT_PRESET_NAME, get_preset, get_preset_names
from common.config_merge import deep_merge_dict, build_colmap_override_dict
from common.colmap_capabilities import append_supported_option


SCRIPT_NAME = "colmap_reconstruction"


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_dirs(root: Path, logger) -> dict[str, Path]:
    paths = {
        "data": root / "data",
        "colmap": root / "data" / "colmap",
        "colmap_images": root / "data" / "colmap" / "images",
        "colmap_sparse": root / "data" / "colmap" / "sparse",
        "logs": root / "logs",
        "scripts": root / "scripts",
        "colmap_install_dir": root / "COLMAP",
        "colmap_exe": root / "COLMAP" / "bin" / "colmap.exe",
        "colmap_bat": root / "COLMAP" / "COLMAP.bat",
    }

    for key in ("data", "colmap", "colmap_images", "colmap_sparse", "logs", "scripts"):
        paths[key].mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s -> %s", key, paths[key])

    return paths


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

    is_bat = str(cmd[0]).lower().endswith(".bat")

    if is_bat:
        cmd_str = subprocess.list2cmdline(cmd)
        wrapped_cmd = ["cmd.exe", "/c", cmd_str]
        logger.debug("Wrapped batch command: %s", quote_cmd(wrapped_cmd))
        logger.debug("Using .bat execution path")

        process = subprocess.Popen(
            wrapped_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
        )
    else:
        logger.debug("Using .exe execution path")
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


def detect_pycolmap(logger):
    try:
        import pycolmap  # type: ignore
        version = getattr(pycolmap, "__version__", "unknown")
        logger.info("Detected pycolmap: version=%s", version)
        return pycolmap
    except ImportError:
        logger.info("pycolmap not available")
        return None
    except Exception as exc:
        logger.warning("pycolmap import failed unexpectedly: %s", exc)
        return None


def detect_colmap_bat(paths: dict[str, Path], logger) -> Path | None:
    bat_path = paths["colmap_bat"]
    if bat_path.exists():
        logger.info("Detected COLMAP.bat: %s", bat_path)
        return bat_path
    logger.info("COLMAP.bat not found at expected path: %s", bat_path)
    return None


def detect_colmap_exe(paths: dict[str, Path], logger) -> Path | None:
    exe_path = paths["colmap_exe"]
    if exe_path.exists():
        logger.info("Detected colmap.exe: %s", exe_path)
        return exe_path
    logger.info("colmap.exe not found at expected path: %s", exe_path)
    return None


def safe_remove_file(path: Path, logger) -> None:
    if path.exists():
        path.unlink()
        logger.debug("Removed file: %s", path)


def safe_remove_sparse_subdirs(sparse_dir: Path, logger) -> int:
    removed = 0
    if not sparse_dir.exists():
        return removed

    for child in sparse_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
            removed += 1
            logger.debug("Removed sparse subdirectory: %s", child)

    return removed


def count_input_images(images_dir: Path) -> int:
    exts = {".jpg", ".jpeg", ".png"}
    return sum(1 for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def build_colmap_cli_commands(
    colmap_cmd: Path,
    database_path: Path,
    image_path: Path,
    sparse_path: Path,
    matcher_name: str,
    camera_model: str,
    single_camera: bool,
    use_gpu: bool,
    max_image_size: int | None,
    logger,
) -> list[tuple[str, list[str]]]:
    single_camera_flag = "1" if single_camera else "0"
    gpu_flag = "1" if use_gpu else "0"

    feature_cmd = [
        str(colmap_cmd),
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--ImageReader.camera_model",
        camera_model,
        "--ImageReader.single_camera",
        single_camera_flag,
    ]

    append_supported_option(
        feature_cmd,
        colmap_cmd,
        "feature_extractor",
        "--SiftExtraction.use_gpu",
        gpu_flag,
        logger=logger,
    )

    if max_image_size is not None:
        append_supported_option(
            feature_cmd,
            colmap_cmd,
            "feature_extractor",
            "--SiftExtraction.max_image_size",
            str(max_image_size),
            logger=logger,
        )

    matcher_cmd = [
        str(colmap_cmd),
        matcher_name,
        "--database_path",
        str(database_path),
    ]

    append_supported_option(
        matcher_cmd,
        colmap_cmd,
        matcher_name,
        "--SiftMatching.use_gpu",
        gpu_flag,
        logger=logger,
    )

    mapper_cmd = [
        str(colmap_cmd),
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--output_path",
        str(sparse_path),
    ]

    return [
        ("feature_extractor", feature_cmd),
        (matcher_name, matcher_cmd),
        ("mapper", mapper_cmd),
    ]


def run_cli_pipeline(
    colmap_cmd: Path,
    database_path: Path,
    image_path: Path,
    sparse_path: Path,
    matcher_name: str,
    camera_model: str,
    single_camera: bool,
    use_gpu: bool,
    max_image_size: int | None,
    logger,
    verbose: bool,
) -> None:
    commands = build_colmap_cli_commands(
        colmap_cmd=colmap_cmd,
        database_path=database_path,
        image_path=image_path,
        sparse_path=sparse_path,
        matcher_name=matcher_name,
        camera_model=camera_model,
        single_camera=single_camera,
        use_gpu=use_gpu,
        max_image_size=max_image_size,
        logger=logger,
    )

    for step_name, cmd in commands:
        logger.info("Starting CLI step: %s", step_name)
        run_command_streaming(cmd, logger, verbose=verbose)
        logger.info("Completed CLI step: %s", step_name)


def run_pycolmap_pipeline(
    pycolmap,
    database_path: Path,
    image_path: Path,
    sparse_path: Path,
    matcher_name: str,
    camera_model: str,
    single_camera: bool,
    use_gpu: bool,
    max_image_size: int | None,
    logger,
) -> None:
    logger.info("Starting pycolmap pipeline")

    feature_options = {
        "camera_model": camera_model,
        "single_camera": single_camera,
        "use_gpu": use_gpu,
    }
    if max_image_size is not None:
        feature_options["max_image_size"] = max_image_size

    logger.debug("pycolmap feature options: %s", feature_options)

    if hasattr(pycolmap, "extract_features"):
        logger.info("pycolmap: extract_features")
        pycolmap.extract_features(
            database_path=str(database_path),
            image_path=str(image_path),
            reader_options=feature_options,
        )
    else:
        raise RuntimeError("pycolmap is installed, but extract_features is unavailable")

    if matcher_name != "sequential_matcher":
        raise RuntimeError(
            f"pycolmap path currently supports sequential_matcher only, not {matcher_name}"
        )

    if hasattr(pycolmap, "match_sequential"):
        logger.info("pycolmap: match_sequential")
        pycolmap.match_sequential(
            database_path=str(database_path),
            sift_matching_options={"use_gpu": use_gpu},
        )
    else:
        raise RuntimeError("pycolmap is installed, but match_sequential is unavailable")

    if hasattr(pycolmap, "incremental_mapping"):
        logger.info("pycolmap: incremental_mapping")
        maps = pycolmap.incremental_mapping(
            database_path=str(database_path),
            image_path=str(image_path),
            output_path=str(sparse_path),
            options={},
        )
        logger.info("pycolmap produced %s sparse model(s)", len(maps) if maps is not None else 0)
    else:
        raise RuntimeError("pycolmap is installed, but incremental_mapping is unavailable")

    logger.info("Completed pycolmap pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run COLMAP sparse reconstruction using presets and overrides."
    )

    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET_NAME,
        choices=get_preset_names(),
        help="COLMAP preset name.",
    )
    parser.add_argument("--images", type=str, default=None)
    parser.add_argument("--database", type=str, default=None)
    parser.add_argument("--sparse", type=str, default=None)
    parser.add_argument("--matcher", type=str, default=None)
    parser.add_argument("--camera-model", type=str, default=None)
    parser.add_argument("--single-camera", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--max-image-size", type=int, default=None)
    parser.add_argument("--force-cli", action="store_true")
    parser.add_argument("--reset", action="store_true")
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
        logger.info("Project folders checked and created if missing")

        preset = get_preset(args.preset)
        colmap_config = deep_merge_dict(
            preset["colmap"],
            build_colmap_override_dict(args),
        )

        image_path = Path(args.images).resolve() if args.images else paths["colmap_images"]
        database_path = Path(args.database).resolve() if args.database else (paths["colmap"] / "database.db")
        sparse_path = Path(args.sparse).resolve() if args.sparse else paths["colmap_sparse"]

        logger.info("Preset: %s", args.preset)
        logger.info("Image path: %s", image_path)
        logger.info("Database path: %s", database_path)
        logger.info("Sparse output path: %s", sparse_path)
        logger.info("COLMAP config: %s", colmap_config)

        if not image_path.exists():
            logger.error("Image path does not exist: %s", image_path)
            return 1

        image_count = count_input_images(image_path)
        logger.info("Detected %s input image(s)", image_count)

        if image_count == 0:
            logger.error("No input images found in: %s", image_path)
            return 1

        sparse_path.mkdir(parents=True, exist_ok=True)

        if args.reset:
            logger.info("Reset requested")
            safe_remove_file(database_path, logger)
            removed_dirs = safe_remove_sparse_subdirs(sparse_path, logger)
            logger.info("Removed %s existing sparse model folder(s)", removed_dirs)

        pycolmap_module = None if args.force_cli else detect_pycolmap(logger)
        colmap_exe = detect_colmap_exe(paths, logger)
        colmap_bat = detect_colmap_bat(paths, logger)

        if pycolmap_module is not None:
            logger.info("Selected backend: pycolmap")
            try:
                run_pycolmap_pipeline(
                    pycolmap=pycolmap_module,
                    database_path=database_path,
                    image_path=image_path,
                    sparse_path=sparse_path,
                    matcher_name=colmap_config["matcher"],
                    camera_model=colmap_config["camera_model"],
                    single_camera=colmap_config["single_camera"],
                    use_gpu=colmap_config["use_gpu"],
                    max_image_size=colmap_config["max_image_size"],
                    logger=logger,
                )
            except Exception as exc:
                logger.warning("pycolmap pipeline failed: %s", exc)
                if colmap_exe is not None:
                    logger.info("Falling back to colmap.exe backend")
                    run_cli_pipeline(
                        colmap_cmd=colmap_exe,
                        database_path=database_path,
                        image_path=image_path,
                        sparse_path=sparse_path,
                        matcher_name=colmap_config["matcher"],
                        camera_model=colmap_config["camera_model"],
                        single_camera=colmap_config["single_camera"],
                        use_gpu=colmap_config["use_gpu"],
                        max_image_size=colmap_config["max_image_size"],
                        logger=logger,
                        verbose=args.verbose,
                    )
                elif colmap_bat is not None:
                    logger.info("Falling back to COLMAP.bat backend")
                    run_cli_pipeline(
                        colmap_cmd=colmap_bat,
                        database_path=database_path,
                        image_path=image_path,
                        sparse_path=sparse_path,
                        matcher_name=colmap_config["matcher"],
                        camera_model=colmap_config["camera_model"],
                        single_camera=colmap_config["single_camera"],
                        use_gpu=colmap_config["use_gpu"],
                        max_image_size=colmap_config["max_image_size"],
                        logger=logger,
                        verbose=args.verbose,
                    )
                else:
                    logger.error("No CLI fallback available after pycolmap failure")
                    return 1

        elif colmap_exe is not None:
            logger.info("Selected backend: colmap.exe")
            run_cli_pipeline(
                colmap_cmd=colmap_exe,
                database_path=database_path,
                image_path=image_path,
                sparse_path=sparse_path,
                matcher_name=colmap_config["matcher"],
                camera_model=colmap_config["camera_model"],
                single_camera=colmap_config["single_camera"],
                use_gpu=colmap_config["use_gpu"],
                max_image_size=colmap_config["max_image_size"],
                logger=logger,
                verbose=args.verbose,
            )
        elif colmap_bat is not None:
            logger.info("Selected backend: COLMAP.bat")
            run_cli_pipeline(
                colmap_cmd=colmap_bat,
                database_path=database_path,
                image_path=image_path,
                sparse_path=sparse_path,
                matcher_name=colmap_config["matcher"],
                camera_model=colmap_config["camera_model"],
                single_camera=colmap_config["single_camera"],
                use_gpu=colmap_config["use_gpu"],
                max_image_size=colmap_config["max_image_size"],
                logger=logger,
                verbose=args.verbose,
            )
        else:
            logger.error("No COLMAP backend available.")
            logger.error("Expected colmap.exe at: %s", paths["colmap_exe"])
            logger.error("Expected COLMAP.bat at: %s", paths["colmap_bat"])
            return 1

        logger.info("COLMAP reconstruction completed successfully")

        model_dirs = [p for p in sparse_path.iterdir() if p.is_dir()] if sparse_path.exists() else []
        logger.info("Sparse model folder count: %s", len(model_dirs))
        for model_dir in model_dirs[:10]:
            logger.info("Sparse model folder: %s", model_dir)

        return 0

    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())