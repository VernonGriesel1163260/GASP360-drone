from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.append(str(Path(__file__).resolve().parent))

from common.logging_utils import setup_logger


SCRIPT_NAME = "run_experiments"


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parent.parent


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(f'"{c}"' if " " in c else c for c in cmd)


def run_command_streaming(
    cmd: list[str],
    logger,
    verbose: bool,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    logger.info("Running command")
    logger.debug("Command: %s", quote_cmd(cmd))
    if cwd:
        logger.debug("Working directory: %s", cwd)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        cwd=str(cwd) if cwd else None,
        env=env,
        bufsize=1,
    )

    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        logger.debug("[SUBPROCESS] %s", line)
        if verbose:
            print(f"[SUBPROCESS] {line}")

    return process.wait()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def experiment_status_path(workspace_root: Path) -> Path:
    return workspace_root / "experiment_status.json"


def load_experiment_status(workspace_root: Path) -> dict[str, Any]:
    path = experiment_status_path(workspace_root)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_experiment_status(workspace_root: Path, payload: dict[str, Any]) -> None:
    write_json(experiment_status_path(workspace_root), payload)


def build_pipeline_command(
    python_exe: str,
    pipeline_script: Path,
    input_video: str,
    global_args: list[str],
    experiment_args: list[str],
    step_from: str | None,
    step_to: str | None,
    verbose: bool,
) -> list[str]:
    cmd = [python_exe, str(pipeline_script)]

    if input_video and "--input-video" not in global_args and "--input-video" not in experiment_args:
        cmd.extend(["--input-video", input_video])

    cmd.extend(global_args)
    cmd.extend(experiment_args)

    if step_from:
        cmd.extend(["--step-from", step_from])
    if step_to:
        cmd.extend(["--step-to", step_to])

    if verbose and "--verbose" not in cmd:
        cmd.append("--verbose")

    return cmd


def build_inspect_command(
    python_exe: str,
    inspect_script: Path,
    inspect_args: list[str],
    verbose: bool,
) -> list[str]:
    cmd = [python_exe, str(inspect_script)]
    cmd.extend(inspect_args)
    if verbose and "--verbose" not in cmd:
        cmd.append("--verbose")
    return cmd


def build_report_command(
    python_exe: str,
    report_script: Path,
    verbose: bool,
) -> list[str]:
    cmd = [python_exe, str(report_script)]
    if verbose:
        cmd.append("--verbose")
    return cmd


def collect_experiment_summary(workspace_root: Path, experiment_id: str) -> dict[str, Any]:
    colmap_dir = workspace_root / "data" / "colmap"
    inspection_path = colmap_dir / "model_inspection.json"
    best_txt = colmap_dir / "best_model.txt"
    failure_reason_path = colmap_dir / "colmap_failure_reason.txt"
    context_path = workspace_root / "workspace_context.json"
    context = read_json_if_exists(context_path) or read_json_if_exists(workspace_root / "data" / "workspace_context.json") or {}
    status = load_experiment_status(workspace_root)

    normalization = context.get("normalization") or {}
    metrics = context.get("metrics") or {}

    payload = {
        "experiment_id": experiment_id,
        "workspace_root": str(workspace_root),
        "selected_model_name": None,
        "selected_model_path": None,
        "registered_images": metrics.get("registered_images"),
        "total_input_images": metrics.get("total_input_images"),
        "registration_ratio": metrics.get("registration_ratio"),
        "points3D": metrics.get("points3D"),
        "observations": metrics.get("observations"),
        "best_model_found": False,
        "failure_reason": None,
        "normalize_selected_pair": ",".join(str(v) for v in (normalization.get("selected_pair") or [])),
        "normalize_resolved_layout": normalization.get("resolved_layout"),
        "normalize_effective_convert_input_format": normalization.get("effective_convert_input_format"),
        "normalize_orientation_signature": normalization.get("orientation_signature"),
        "normalize_rotate_a": ((normalization.get("transforms") or {}).get("stream_a") or {}).get("rotate"),
        "normalize_flip_h_a": ((normalization.get("transforms") or {}).get("stream_a") or {}).get("flip_h"),
        "normalize_flip_v_a": ((normalization.get("transforms") or {}).get("stream_a") or {}).get("flip_v"),
        "normalize_rotate_b": ((normalization.get("transforms") or {}).get("stream_b") or {}).get("rotate"),
        "normalize_flip_h_b": ((normalization.get("transforms") or {}).get("stream_b") or {}).get("flip_h"),
        "normalize_flip_v_b": ((normalization.get("transforms") or {}).get("stream_b") or {}).get("flip_v"),
    }

    if failure_reason_path.exists():
        failure_reason = failure_reason_path.read_text(encoding="utf-8").strip()
        if failure_reason:
            payload["failure_reason"] = failure_reason

    inspection = read_json_if_exists(inspection_path)
    if inspection and inspection.get("best_model"):
        best = inspection["best_model"]
        payload["selected_model_name"] = best.get("name")
        payload["selected_model_path"] = best.get("path")
        payload["registered_images"] = best.get("registered_images", payload["registered_images"])
        payload["total_input_images"] = best.get("total_input_images", payload["total_input_images"])
        payload["registration_ratio"] = best.get("registration_ratio", payload["registration_ratio"])
        payload["points3D"] = best.get("points3D", payload["points3D"])
        payload["observations"] = best.get("observations", payload["observations"])
        payload["best_model_found"] = True
        payload["failure_reason"] = None
        return payload

    if best_txt.exists():
        lines = [line.strip() for line in best_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) >= 2:
            payload["selected_model_name"] = lines[0]
            payload["selected_model_path"] = lines[1]
            payload["best_model_found"] = True
            payload["failure_reason"] = None

    if payload["failure_reason"] is None:
        state = status.get("state")
        if isinstance(state, str) and state.startswith("failed_"):
            exit_code = status.get("last_exit_code")
            if exit_code is None:
                payload["failure_reason"] = state
            else:
                payload["failure_reason"] = f"{state} (exit_code={exit_code})"

    return payload


def write_master_summary(base_output_root: Path, rows: list[dict[str, Any]]) -> None:
    json_path = base_output_root / "experiments_summary.json"
    csv_path = base_output_root / "experiments_summary.csv"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = [
        "experiment_id",
        "workspace_root",
        "selected_model_name",
        "selected_model_path",
        "registered_images",
        "total_input_images",
        "registration_ratio",
        "points3D",
        "observations",
        "best_model_found",
        "failure_reason",
        "normalize_selected_pair",
        "normalize_resolved_layout",
        "normalize_effective_convert_input_format",
        "normalize_orientation_signature",
        "normalize_rotate_a",
        "normalize_flip_h_a",
        "normalize_flip_v_a",
        "normalize_rotate_b",
        "normalize_flip_h_b",
        "normalize_flip_v_b",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a resumable experiment matrix using isolated workspaces.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--experiment", type=str, default=None, help="Run only one experiment id.")
    parser.add_argument("--step-from", type=str, default=None)
    parser.add_argument("--step-to", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip experiments marked done.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root_from_script()
    logger, run_log_path, latest_log_path = setup_logger(SCRIPT_NAME, verbose=args.verbose, workspace_root=root)

    try:
        config_path = Path(args.config).expanduser().resolve()
        cfg = load_yaml(config_path)

        global_cfg = cfg.get("global", {})
        experiments = cfg.get("experiments", [])

        base_output_root = Path(global_cfg["base_output_root"]).expanduser().resolve()
        ensure_dir(base_output_root)

        python_exe = global_cfg.get("python_executable", sys.executable)
        input_video = global_cfg.get("input_video", "")
        input_video_path = Path(input_video).expanduser().resolve()
        if input_video and not input_video_path.exists():
            logger.error("Configured input video does not exist: %s", input_video_path)
            return 1

        pipeline_args = list(global_cfg.get("pipeline_args", []))
        inspect_args = list(global_cfg.get("inspect_args", ["--summary-json", "--promote-best"]))

        pipeline_script = root / "scripts" / "pipeline.py"
        inspect_script = root / "scripts" / "inspect_colmap_models.py"
        report_script = root / "scripts" / "pipeline_report.py"

        logger.info("Config: %s", config_path)
        logger.info("Base output root: %s", base_output_root)
        logger.info("Experiments in config: %s", len(experiments))

        all_rows: list[dict[str, Any]] = []

        for exp in experiments:
            exp_id = exp["id"]
            if args.experiment and exp_id != args.experiment:
                continue

            workspace_root = base_output_root / exp_id
            ensure_dir(workspace_root)

            status = load_experiment_status(workspace_root)
            if args.resume and status.get("state") == "done":
                logger.info("Skipping completed experiment: %s", exp_id)
                all_rows.append(collect_experiment_summary(workspace_root, exp_id))
                continue

            logger.info("=== Experiment: %s ===", exp_id)

            env = os.environ.copy()
            env["GASP_WORKSPACE_ROOT"] = str(workspace_root)

            exp_pipeline_args = list(exp.get("pipeline_args", []))

            pipeline_cmd = build_pipeline_command(
                python_exe=python_exe,
                pipeline_script=pipeline_script,
                input_video=input_video,
                global_args=pipeline_args,
                experiment_args=exp_pipeline_args,
                step_from=args.step_from,
                step_to=args.step_to,
                verbose=args.verbose,
            )

            save_experiment_status(
                workspace_root,
                {
                    "experiment_id": exp_id,
                    "state": "running_pipeline",
                    "updated_at": datetime.now().isoformat(),
                    "workspace_root": str(workspace_root),
                },
            )

            exit_code = run_command_streaming(
                pipeline_cmd,
                logger,
                verbose=args.verbose,
                cwd=root,
                env=env,
            )
            if exit_code != 0:
                save_experiment_status(
                    workspace_root,
                    {
                        "experiment_id": exp_id,
                        "state": "failed_pipeline",
                        "updated_at": datetime.now().isoformat(),
                        "last_exit_code": exit_code,
                        "workspace_root": str(workspace_root),
                    },
                )
                logger.error("Experiment failed during pipeline: %s", exp_id)
                continue

            inspect_cmd = build_inspect_command(
                python_exe=python_exe,
                inspect_script=inspect_script,
                inspect_args=inspect_args,
                verbose=args.verbose,
            )

            save_experiment_status(
                workspace_root,
                {
                    "experiment_id": exp_id,
                    "state": "running_inspect",
                    "updated_at": datetime.now().isoformat(),
                    "workspace_root": str(workspace_root),
                },
            )

            exit_code = run_command_streaming(
                inspect_cmd,
                logger,
                verbose=args.verbose,
                cwd=root,
                env=env,
            )
            if exit_code != 0:
                save_experiment_status(
                    workspace_root,
                    {
                        "experiment_id": exp_id,
                        "state": "failed_inspect",
                        "updated_at": datetime.now().isoformat(),
                        "last_exit_code": exit_code,
                        "workspace_root": str(workspace_root),
                    },
                )
                logger.error("Experiment failed during inspection: %s", exp_id)
                continue

            report_cmd = build_report_command(
                python_exe=python_exe,
                report_script=report_script,
                verbose=args.verbose,
            )

            save_experiment_status(
                workspace_root,
                {
                    "experiment_id": exp_id,
                    "state": "running_report",
                    "updated_at": datetime.now().isoformat(),
                    "workspace_root": str(workspace_root),
                },
            )

            exit_code = run_command_streaming(
                report_cmd,
                logger,
                verbose=args.verbose,
                cwd=root,
                env=env,
            )
            if exit_code != 0:
                save_experiment_status(
                    workspace_root,
                    {
                        "experiment_id": exp_id,
                        "state": "failed_report",
                        "updated_at": datetime.now().isoformat(),
                        "last_exit_code": exit_code,
                        "workspace_root": str(workspace_root),
                    },
                )
                logger.error("Experiment failed during report: %s", exp_id)
                continue

            save_experiment_status(
                workspace_root,
                {
                    "experiment_id": exp_id,
                    "state": "done",
                    "updated_at": datetime.now().isoformat(),
                    "workspace_root": str(workspace_root),
                },
            )

            row = collect_experiment_summary(workspace_root, exp_id)
            all_rows.append(row)
            write_master_summary(base_output_root, all_rows)

        final_rows: list[dict[str, Any]] = []
        for exp in experiments:
            exp_id = exp["id"]
            if args.experiment and exp_id != args.experiment:
                continue
            workspace_root = base_output_root / exp_id
            final_rows.append(collect_experiment_summary(workspace_root, exp_id))

        write_master_summary(base_output_root, final_rows)
        logger.info("Experiment matrix completed")
        return 0

    except Exception as exc:
        logger.exception("Fatal experiment-runner error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())