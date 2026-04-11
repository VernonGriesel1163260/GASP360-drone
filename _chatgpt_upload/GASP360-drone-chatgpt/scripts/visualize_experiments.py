from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def plot_bar(labels: list[str], values: list[float], title: str, ylabel: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(rows: list[dict[str, Any]], out_path: Path) -> None:
    lines = [
        "# Experiment Summary",
        "",
        "| Experiment | Best Model Found | Registered Images | Total Input Images | Registration Ratio | Points3D | Observations |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        experiment_id = row.get("experiment_id", "")
        best_model_found = row.get("best_model_found", "")
        registered_images = row.get("registered_images", "")
        total_input_images = row.get("total_input_images", "")
        registration_ratio = row.get("registration_ratio", "")

        if registration_ratio not in (None, ""):
            try:
                registration_ratio = f"{float(registration_ratio) * 100.0:.2f}%"
            except Exception:
                pass

        points3d = row.get("points3D", "")
        observations = row.get("observations", "")

        lines.append(
            f"| {experiment_id} | {best_model_found} | {registered_images} | "
            f"{total_input_images} | {registration_ratio} | {points3d} | {observations} |"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate experiment visualisations from experiments_summary.csv")
    parser.add_argument("--summary-csv", type=str, required=True, help="Path to experiments_summary.csv")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include experiments where best_model_found is false",
    )
    args = parser.parse_args()

    csv_path = Path(args.summary_csv).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)

    if not args.include_failed:
        rows = [r for r in rows if is_true(r.get("best_model_found"))]

    if not rows:
        print("No experiment rows available for plotting.")
        return 1

    labels = [str(r["experiment_id"]) for r in rows]
    registration_ratio_pct = [to_float(r.get("registration_ratio")) * 100.0 for r in rows]
    registered_images = [to_int(r.get("registered_images")) for r in rows]
    points3d = [to_int(r.get("points3D")) for r in rows]
    observations = [to_int(r.get("observations")) for r in rows]

    plot_bar(
        labels,
        registration_ratio_pct,
        "Registration Ratio by Experiment",
        "Registration Ratio (%)",
        out_dir / "registration_ratio.png",
    )

    plot_bar(
        labels,
        registered_images,
        "Registered Images by Experiment",
        "Registered Images",
        out_dir / "registered_images.png",
    )

    plot_bar(
        labels,
        points3d,
        "Points3D by Experiment",
        "Points3D",
        out_dir / "points3d.png",
    )

    plot_bar(
        labels,
        observations,
        "Observations by Experiment",
        "Observations",
        out_dir / "observations.png",
    )

    write_markdown_summary(rows, out_dir / "summary.md")

    print(f"Saved outputs to: {out_dir}")
    print("Generated:")
    print(" - registration_ratio.png")
    print(" - registered_images.png")
    print(" - points3d.png")
    print(" - observations.png")
    print(" - summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())