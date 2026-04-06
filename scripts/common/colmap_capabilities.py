from __future__ import annotations

import subprocess
from pathlib import Path


def _build_help_probe_command(colmap_cmd: Path, subcommand: str) -> list[str]:
    base_cmd = [str(colmap_cmd), subcommand, "-h"]
    if str(colmap_cmd).lower().endswith(".bat"):
        return ["cmd.exe", "/c", subprocess.list2cmdline(base_cmd)]
    return base_cmd


def get_colmap_help_text(colmap_cmd: Path, subcommand: str) -> str:
    try:
        result = subprocess.run(
            _build_help_probe_command(colmap_cmd, subcommand),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            check=False,
        )
    except OSError:
        return ""

    return (result.stdout or "") + "\n" + (result.stderr or "")


def command_supports_option(colmap_cmd: Path, subcommand: str, option_name: str) -> bool:
    help_text = get_colmap_help_text(colmap_cmd, subcommand)
    return option_name in help_text


def append_supported_option(
    cmd: list[str],
    colmap_cmd: Path,
    subcommand: str,
    option_name: str,
    option_value: str | None,
    logger=None,
) -> bool:
    if not command_supports_option(colmap_cmd, subcommand, option_name):
        if logger:
            logger.debug(
                "Skipping unsupported option for %s: %s",
                subcommand,
                option_name,
            )
        return False

    cmd.append(option_name)
    if option_value is not None:
        cmd.append(option_value)

    if logger:
        logger.debug(
            "Added supported option for %s: %s=%s",
            subcommand,
            option_name,
            option_value,
        )
    return True
