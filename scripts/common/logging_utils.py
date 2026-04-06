from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_log_dir(script_name: str) -> Path:
    log_dir = get_project_root() / "logs" / script_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_logger(script_name: str, verbose: bool = False) -> tuple[logging.Logger, Path, Path]:
    log_dir = ensure_log_dir(script_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = log_dir / f"{script_name}_{timestamp}.log"
    latest_log_path = log_dir / f"{script_name}_latest.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console_level = logging.DEBUG if verbose else logging.INFO

    console_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(run_log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    latest_handler = logging.FileHandler(latest_log_path, encoding="utf-8", mode="w")
    latest_handler.setLevel(logging.DEBUG)
    latest_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(latest_handler)

    logger.debug("Logger initialised")
    logger.debug("Verbose mode: %s", verbose)
    logger.debug("Run log path: %s", run_log_path)
    logger.debug("Latest log path: %s", latest_log_path)

    return logger, run_log_path, latest_log_path