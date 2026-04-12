from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

# Make ./scripts importable when running:
# python .\scripts\preprocess_input_video.py
sys.path.append(str(Path(__file__).resolve().parent))

from common.logging_utils import setup_logger
from common.workspace import resolve_code_root, resolve_workspace_root


SCRIPT_NAME = "preprocess_input_video"
PREPROCESS_METADATA_FILENAME = "_preprocess_metadata.json"

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mts", ".m2ts",
    ".ts", ".wmv", ".flv", ".mpg", ".mpeg", ".osv", ".insv", ".360",
}

FRAME_FORMAT_ALIASES = {
    "equirect": "equirect",
    "equirectangular": "equirect",
    "fisheye": "fisheye",
    "fishere": "fisheye",
    "dfisheye": "dfisheye",
    "dual-fisheye": "dfisheye",
    "dual_fisheye": "dfisheye",
    "flat": "flat",
    "plain": "flat",
    "rectilinear": "flat",
    "cubemap": "cubemap",
    "c3x2": "c3x2",
    "c6x1": "c6x1",
    "c1x6": "c1x6",
}

RECOMMENDED_STRATEGIES = {
    "single_stream",
    "single_stream_best_of_n",
    "extract_both_then_stitch",
    "manual_review_required",
}

CLASS_DUPLICATE = "duplicate_or_near_duplicate"
CLASS_TRANSFORMED = "same_scene_transformed"
CLASS_COMPLEMENTARY = "complementary_or_distinct"
CLASS_UNCERTAIN = "uncertain"


def code_root_from_script() -> Path:
    return resolve_code_root(__file__)


def workspace_root_from_script() -> Path:
    return resolve_workspace_root(caller_file=__file__)


def ensure_dirs(code_root: Path, workspace_root: Path, logger) -> dict[str, Path]:
    paths = {
        "code_root": code_root,
        "workspace_root": workspace_root,
        "data": workspace_root / "data",
        "input_video": workspace_root / "data" / "input_video",
        "logs": workspace_root / "logs",
        "preprocess_root": workspace_root / "data" / "input_video" / "_preprocess",
        "preprocess_samples": workspace_root / "data" / "input_video" / "_preprocess" / "samples",
        "preprocess_metadata": workspace_root / "data" / "input_video" / PREPROCESS_METADATA_FILENAME,
    }

    for key in ("data", "input_video", "logs", "preprocess_root", "preprocess_samples"):
        paths[key].mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s -> %s", key, paths[key])

    return paths


def append_optional_value(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(f'"{c}"' if " " in c else c for c in cmd)


def run_command(cmd: list[str], logger) -> subprocess.CompletedProcess:
    logger.debug("Running command: %s", quote_cmd(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        check=False,
    )
    if result.stdout.strip():
        logger.debug("STDOUT: %s", result.stdout.strip())
    if result.stderr.strip():
        logger.debug("STDERR: %s", result.stderr.strip())
    return result


def find_ffmpeg(code_root: Path) -> Path:
    candidates = [
        code_root / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
        code_root / "tools" / "ffmpeg" / "bin" / "ffmpeg",
        code_root / "tools" / "ffmpeg.exe",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()

    which_name = "ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"
    resolved = shutil.which(which_name) or shutil.which("ffmpeg")
    if resolved:
        return Path(resolved).resolve()

    raise FileNotFoundError(
        "FFmpeg was not found. Expected tools/ffmpeg/bin/ffmpeg(.exe), tools/ffmpeg.exe, or ffmpeg on PATH."
    )


def find_ffprobe(code_root: Path) -> Path:
    candidates = [
        code_root / "tools" / "ffmpeg" / "bin" / "ffprobe.exe",
        code_root / "tools" / "ffmpeg" / "bin" / "ffprobe",
        code_root / "tools" / "ffprobe.exe",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()

    which_name = "ffprobe.exe" if sys.platform.startswith("win") else "ffprobe"
    resolved = shutil.which(which_name) or shutil.which("ffprobe")
    if resolved:
        return Path(resolved).resolve()

    raise FileNotFoundError(
        "ffprobe was not found. Expected tools/ffmpeg/bin/ffprobe(.exe), tools/ffprobe.exe, or ffprobe on PATH."
    )


def list_supported_videos(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def resolve_input_video(input_arg: str | None, input_dir: Path) -> Path:
    if input_arg:
        candidate = Path(input_arg).expanduser()
        if candidate.is_absolute():
            return candidate.resolve(strict=False)
        return (Path.cwd() / candidate).resolve(strict=False)

    available = list_supported_videos(input_dir)
    if not available:
        raise FileNotFoundError(
            f"No supported videos found in {input_dir}. Supported extensions: {', '.join(sorted(VIDEO_EXTENSIONS))}"
        )

    return available[0].resolve()


def parse_fraction_list(raw_values: list[float] | None) -> list[float]:
    if raw_values:
        cleaned = []
        for value in raw_values:
            if not 0.0 < value < 1.0:
                raise ValueError(f"Sample position fractions must be between 0 and 1 (exclusive), got {value}")
            cleaned.append(float(value))
        return sorted(set(cleaned))
    return []


def build_sample_fractions(sample_count: int, sample_positions: list[float]) -> list[float]:
    if sample_positions:
        return sample_positions

    if sample_count <= 1:
        return [0.50]

    start = 0.20
    end = 0.80
    if sample_count == 2:
        return [0.33, 0.66]

    step = (end - start) / (sample_count - 1)
    return [round(start + (i * step), 4) for i in range(sample_count)]


def ffprobe_media_info(ffprobe_path: Path, input_video: Path, logger) -> dict:
    show_entries = (
        "format=filename,duration,size,bit_rate:"
        "stream=index,codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,duration,bit_rate:"
        "stream_disposition=attached_pic,default:"
        "stream_tags=handler_name,comment,creation_time,language"
    )
    cmd = [
        str(ffprobe_path),
        "-v", "error",
        "-show_entries",
        show_entries,
        "-of", "json",
        str(input_video),
    ]
    result = run_command(cmd, logger)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {input_video}: {result.stderr.strip()}")

    try:
        return json.loads(result.stdout)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse ffprobe JSON for {input_video}: {exc}") from exc


def parse_fraction(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    try:
        return float(Fraction(value))
    except Exception:
        return 0.0


def is_attached_picture_stream(stream: dict) -> bool:
    disposition = stream.get("disposition") or {}
    if disposition.get("attached_pic") == 1:
        return True

    codec_name = (stream.get("codec_name") or "").lower()
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    avg_fps = parse_fraction(stream.get("avg_frame_rate"))
    r_fps = parse_fraction(stream.get("r_frame_rate"))
    tags = stream.get("tags") or {}
    handler_name = (tags.get("handler_name") or "").lower()

    # Common thumbnail / attached preview heuristics
    if codec_name == "mjpeg" and avg_fps == 0.0:
        return True

    if codec_name == "mjpeg" and r_fps >= 1000 and avg_fps == 0.0:
        return True

    if codec_name == "mjpeg" and width <= 1280 and height <= 720 and "videohandler" not in handler_name:
        return True

    return False


def is_real_candidate_video_stream(stream: dict) -> bool:
    if (stream.get("codec_type") or "").lower() != "video":
        return False

    if is_attached_picture_stream(stream):
        return False

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        return False

    return True


def summarize_streams(ffprobe_payload: dict, logger) -> tuple[list[dict], list[dict]]:
    streams = ffprobe_payload.get("streams") or []
    all_video_streams: list[dict] = []

    for ordinal, stream_payload in enumerate(streams):
        if (stream_payload.get("codec_type") or "").lower() != "video":
            continue

        tags = stream_payload.get("tags") or {}

        stream_summary = {
            "stream_index": stream_payload.get("index"),
            "ordinal": len(all_video_streams),
            "video_ordinal": len(all_video_streams),
            "codec_name": stream_payload.get("codec_name"),
            "width": stream_payload.get("width"),
            "height": stream_payload.get("height"),
            "avg_frame_rate": parse_fraction(stream_payload.get("avg_frame_rate")),
            "r_frame_rate": parse_fraction(stream_payload.get("r_frame_rate")),
            "duration": float(stream_payload["duration"]) if stream_payload.get("duration") not in (None, "N/A") else None,
            "bit_rate": int(stream_payload["bit_rate"]) if stream_payload.get("bit_rate") not in (None, "N/A") else None,
            "handler_name": tags.get("handler_name"),
            "comment": tags.get("comment"),
            "language": tags.get("language"),
            "attached_pic": is_attached_picture_stream(stream_payload),
            "raw": stream_payload,
        }

        all_video_streams.append(stream_summary)

    candidate_streams = [s for s in all_video_streams if is_real_candidate_video_stream(s["raw"])]
    excluded_video_streams = [s for s in all_video_streams if not is_real_candidate_video_stream(s["raw"])]

    logger.info("Video streams detected: %s", len(all_video_streams))
    logger.info("Candidate non-attached video streams: %s", len(candidate_streams))

    for stream in candidate_streams:
        tags = stream["raw"].get("tags") or {}
        logger.info(
            "Video stream idx=%s ordinal=%s size=%sx%s codec=%s attached_pic=%s handler=%s comment=%s",
            stream["stream_index"],
            stream["ordinal"],
            stream["width"],
            stream["height"],
            stream["codec_name"],
            is_attached_picture_stream(stream["raw"]),
            tags.get("handler_name"),
            tags.get("comment"),
        )

    for stream in excluded_video_streams:
        tags = stream["raw"].get("tags") or {}
        logger.info(
            "Excluded video stream idx=%s size=%sx%s codec=%s avg_frame_rate=%s r_frame_rate=%s handler=%s",
            stream["stream_index"],
            stream["width"],
            stream["height"],
            stream["codec_name"],
            stream["raw"].get("avg_frame_rate"),
            stream["raw"].get("r_frame_rate"),
            tags.get("handler_name"),
        )

    return all_video_streams, candidate_streams


def timestamp_seconds(duration: float, fraction: float) -> float:
    return max(0.0, min(duration, duration * fraction))


def extract_sample_frame(
    ffmpeg_path: Path,
    input_video: Path,
    stream_index: int,
    timestamp_sec: float,
    output_path: Path,
    logger,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{timestamp_sec:.6f}",
        "-i", str(input_video),
        "-map", f"0:{stream_index}",
        "-frames:v", "1",
        "-q:v", "2",
        str(output_path),
    ]
    result = run_command(cmd, logger)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed extracting sample frame for stream {stream_index} at {timestamp_sec:.3f}s: {result.stderr.strip()}"
        )
    if not output_path.exists():
        raise RuntimeError(f"Expected sample frame was not written: {output_path}")


def load_luma_preview(image_path: Path, size: int = 128) -> tuple[np.ndarray, int, int]:
    with Image.open(image_path) as image:
        width, height = image.size
        preview = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(preview, dtype=np.float32) / 255.0
    return arr, width, height


def region_mean(arr: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> float:
    h, w = arr.shape
    xs0 = max(0, min(w - 1, int(round(x0 * w))))
    xs1 = max(xs0 + 1, min(w, int(round(x1 * w))))
    ys0 = max(0, min(h - 1, int(round(y0 * h))))
    ys1 = max(ys0 + 1, min(h, int(round(y1 * h))))
    return float(arr[ys0:ys1, xs0:xs1].mean())


def infer_frame_format_from_image(image_path: Path) -> dict:
    arr, width, height = load_luma_preview(image_path, size=96)
    ratio = width / max(height, 1)

    corner_mean = float(np.mean([
        region_mean(arr, 0.00, 0.16, 0.00, 0.16),
        region_mean(arr, 0.84, 1.00, 0.00, 0.16),
        region_mean(arr, 0.00, 0.16, 0.84, 1.00),
        region_mean(arr, 0.84, 1.00, 0.84, 1.00),
    ]))
    center_mean = region_mean(arr, 0.40, 0.60, 0.40, 0.60)
    top_mid_mean = region_mean(arr, 0.40, 0.60, 0.00, 0.14)
    bottom_mid_mean = region_mean(arr, 0.40, 0.60, 0.86, 1.00)
    left_lens_mean = region_mean(arr, 0.12, 0.38, 0.24, 0.76)
    right_lens_mean = region_mean(arr, 0.62, 0.88, 0.24, 0.76)

    reasons = [
        f"size={width}x{height}",
        f"aspect_ratio={ratio:.3f}",
        f"corner_mean={corner_mean:.3f}",
        f"center_mean={center_mean:.3f}",
        f"top_mid_mean={top_mid_mean:.3f}",
        f"bottom_mid_mean={bottom_mid_mean:.3f}",
        f"left_lens_mean={left_lens_mean:.3f}",
        f"right_lens_mean={right_lens_mean:.3f}",
    ]

    if 1.45 <= ratio <= 1.55:
        reasons.append("Aspect ratio is close to 3:2, which often matches a 3x2 cubemap layout.")
        return {"format": "c3x2", "confidence": 0.92, "reasons": reasons}
    if 5.5 <= ratio <= 6.5:
        reasons.append("Aspect ratio is close to 6:1, which often matches a 6x1 cubemap layout.")
        return {"format": "c6x1", "confidence": 0.92, "reasons": reasons}
    if 0.14 <= ratio <= 0.19:
        reasons.append("Aspect ratio is close to 1:6, which often matches a 1x6 cubemap layout.")
        return {"format": "c1x6", "confidence": 0.92, "reasons": reasons}

    if (
        1.70 <= ratio <= 2.30
        and corner_mean < 0.12
        and top_mid_mean < 0.18
        and bottom_mid_mean < 0.18
        and left_lens_mean > 0.20
        and right_lens_mean > 0.20
    ):
        reasons.append("Two bright lens regions with dark corners/top/bottom suggest a side-by-side dual-fisheye frame.")
        return {"format": "dfisheye", "confidence": 0.86, "reasons": reasons}

    if (
        0.80 <= ratio <= 1.25
        and corner_mean < 0.12
        and center_mean > corner_mean + 0.12
    ):
        reasons.append("Near-square frame with dark corners and a bright center suggests a single fisheye frame.")
        return {"format": "fisheye", "confidence": 0.82, "reasons": reasons}

    if 1.70 <= ratio <= 2.30 and corner_mean > 0.10:
        reasons.append("2:1 frame with image content present in the corners suggests an equirectangular panorama.")
        return {"format": "equirect", "confidence": 0.78, "reasons": reasons}

    reasons.append("Falling back to flat because the frame does not strongly match the common 360 layouts.")
    return {"format": "flat", "confidence": 0.55, "reasons": reasons}


def dhash_bits(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        thumb = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    arr = np.asarray(thumb, dtype=np.float32)
    return (arr[:, 1:] > arr[:, :-1]).astype(np.uint8).flatten()


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    a_std = float(a_flat.std())
    b_std = float(b_flat.std())
    if a_std < 1e-8 and b_std < 1e-8:
        return 1.0
    if a_std < 1e-8 or b_std < 1e-8:
        return 0.0
    corr = float(np.corrcoef(a_flat, b_flat)[0, 1])
    if math.isnan(corr):
        return 0.0
    return corr


def compare_frame_pair(image_a: Path, image_b: Path) -> dict:
    arr_a, width_a, height_a = load_luma_preview(image_a, size=128)
    arr_b, width_b, height_b = load_luma_preview(image_b, size=128)

    mae = float(np.mean(np.abs(arr_a - arr_b)))
    corr = safe_corrcoef(arr_a, arr_b)
    hamming_distance = int(np.count_nonzero(dhash_bits(image_a) != dhash_bits(image_b)))
    dhash_similarity = 1.0 - (hamming_distance / 64.0)

    return {
        "image_a": image_a.name,
        "image_b": image_b.name,
        "width_a": width_a,
        "height_a": height_a,
        "width_b": width_b,
        "height_b": height_b,
        "mean_absolute_difference": mae,
        "correlation": corr,
        "dhash_hamming_distance": hamming_distance,
        "dhash_similarity": dhash_similarity,
    }


def aggregate_pairwise_metrics(frame_pair_metrics: list[dict]) -> dict:
    if not frame_pair_metrics:
        return {
            "sample_count": 0,
            "mean_absolute_difference": None,
            "correlation": None,
            "dhash_hamming_distance": None,
            "dhash_similarity": None,
        }

    def avg(key: str) -> float:
        return float(sum(metric[key] for metric in frame_pair_metrics) / len(frame_pair_metrics))

    return {
        "sample_count": len(frame_pair_metrics),
        "mean_absolute_difference": avg("mean_absolute_difference"),
        "correlation": avg("correlation"),
        "dhash_hamming_distance": avg("dhash_hamming_distance"),
        "dhash_similarity": avg("dhash_similarity"),
    }


def classify_stream_pair(aggregate_metrics: dict) -> tuple[str, float, list[str]]:
    mae = aggregate_metrics["mean_absolute_difference"]
    corr = aggregate_metrics["correlation"]
    hamming = aggregate_metrics["dhash_hamming_distance"]
    reasons = [
        f"mean_absolute_difference={mae:.4f}",
        f"correlation={corr:.4f}",
        f"dhash_hamming_distance={hamming:.2f}",
    ]

    if mae <= 0.08 and corr >= 0.95 and hamming <= 8:
        reasons.append("Streams are near-duplicates across the sampled timestamps.")
        return CLASS_DUPLICATE, 0.93, reasons

    if mae <= 0.18 and corr >= 0.75 and hamming <= 20:
        reasons.append("Streams look related but not identical, suggesting the same scene with a transformation such as stabilisation or crop.")
        return CLASS_TRANSFORMED, 0.78, reasons

    if mae >= 0.28 or corr <= 0.45 or hamming >= 26:
        reasons.append("Streams differ strongly across the sampled timestamps, suggesting complementary lenses, opposing views, or otherwise distinct imagery.")
        return CLASS_COMPLEMENTARY, 0.84, reasons

    reasons.append("Metrics sit between the common thresholds, so the relationship is uncertain.")
    return CLASS_UNCERTAIN, 0.55, reasons


def normalize_frame_format(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = FRAME_FORMAT_ALIASES.get(value.strip().lower())
    if normalized is None:
        valid = ", ".join(sorted(FRAME_FORMAT_ALIASES))
        raise ValueError(f"Unsupported frame format '{value}'. Valid values include: {valid}")
    return normalized


def choose_preferred_stream(candidate_streams: list[dict]) -> dict | None:
    if not candidate_streams:
        return None

    def ranking_key(stream: dict) -> tuple:
        pixel_count = int((stream.get("width") or 0) * (stream.get("height") or 0))
        bit_rate = int(stream.get("bit_rate") or 0)
        fps = float(stream.get("avg_frame_rate") or 0.0)
        return (pixel_count, bit_rate, fps)

    return max(candidate_streams, key=ranking_key)


def build_strategy_recommendation(
    candidate_streams: list[dict],
    pairwise_results: list[dict],
    forced_primary_stream_index: int | None,
    forced_frame_format: str | None,
    forced_strategy: str | None,
) -> dict:
    recommendation = {
        "recommended_strategy": None,
        "recommended_primary_stream_index": None,
        "recommended_frame_format": forced_frame_format,
        "decision_source": "automatic",
        "reasons": [],
    }

    if forced_strategy is not None:
        recommendation["recommended_strategy"] = forced_strategy
        recommendation["decision_source"] = "forced_strategy"
        recommendation["reasons"].append(f"Strategy was forced by CLI override: {forced_strategy}")

    if forced_primary_stream_index is not None:
        recommendation["recommended_primary_stream_index"] = forced_primary_stream_index
        recommendation["decision_source"] = "forced_primary_stream"
        recommendation["reasons"].append(f"Primary stream was forced by CLI override: {forced_primary_stream_index}")
        if recommendation["recommended_strategy"] is None:
            recommendation["recommended_strategy"] = "single_stream"

    if recommendation["recommended_strategy"] is not None:
        return recommendation

    if len(candidate_streams) == 0:
        recommendation["recommended_strategy"] = "manual_review_required"
        recommendation["reasons"].append("No usable non-attached video streams were found.")
        return recommendation

    if len(candidate_streams) == 1:
        stream = candidate_streams[0]
        recommendation["recommended_strategy"] = "single_stream"
        recommendation["recommended_primary_stream_index"] = stream["stream_index"]
        recommendation["reasons"].append("Only one usable video stream was found.")
        return recommendation

    if pairwise_results:
        best_pair = max(pairwise_results, key=lambda item: item["confidence"])
        relation = best_pair["classification"]

        if relation == CLASS_DUPLICATE:
            preferred = choose_preferred_stream(candidate_streams)
            recommendation["recommended_strategy"] = "single_stream_best_of_n"
            recommendation["recommended_primary_stream_index"] = preferred["stream_index"] if preferred else None
            recommendation["reasons"].append(
                f"At least one stream pair looks near-duplicate, so using one best stream is the lowest-complexity option."
            )
            recommendation["reasons"].append(
                f"Best-supported pair: stream {best_pair['stream_a']} vs stream {best_pair['stream_b']}."
            )
            return recommendation

        if relation == CLASS_TRANSFORMED:
            preferred = choose_preferred_stream(candidate_streams)
            recommendation["recommended_strategy"] = "single_stream_best_of_n"
            recommendation["recommended_primary_stream_index"] = preferred["stream_index"] if preferred else None
            recommendation["reasons"].append(
                "The sampled streams appear related but transformed, so choosing a single best stream is safer than stitching until the geometry is proven."
            )
            recommendation["reasons"].append(
                f"Best-supported pair: stream {best_pair['stream_a']} vs stream {best_pair['stream_b']}."
            )
            return recommendation

        if relation == CLASS_COMPLEMENTARY:
            recommendation["recommended_strategy"] = "extract_both_then_stitch"
            recommendation["reasons"].append(
                "The sampled streams differ strongly, which often means complementary lenses or distinct coverage."
            )
            recommendation["reasons"].append(
                f"Best-supported pair: stream {best_pair['stream_a']} vs stream {best_pair['stream_b']}."
            )
            return recommendation

    recommendation["recommended_strategy"] = "manual_review_required"
    recommendation["reasons"].append("Multiple candidate streams were found, but the comparison was not decisive enough for an automatic recommendation.")
    return recommendation


def serialize_timestamp_label(timestamp_sec: float) -> str:
    milliseconds = int(round(timestamp_sec * 1000.0))
    return f"t{milliseconds:08d}ms"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe a video container, extract a tiny sample from each candidate stream, compare the streams, and write a preprocess sidecar."
    )
    parser.add_argument("--input", type=str, default=None, help="Input video path. Defaults to the first supported video in data/input_video.")
    parser.add_argument(
        "--mode",
        choices=["report-only", "auto"],
        default="report-only",
        help="report-only writes metadata and recommendations only. auto currently also writes metadata only, but marks the intent for future pipeline wiring.",
    )
    parser.add_argument("--sample-count", type=int, default=3, help="Number of evenly spaced timestamps to sample per stream when --sample-positions is not provided.")
    parser.add_argument(
        "--sample-positions",
        nargs="*",
        type=float,
        default=None,
        help="Optional fractions between 0 and 1, such as 0.2 0.5 0.8, used instead of --sample-count.",
    )
    parser.add_argument(
        "--force-primary-stream-index",
        type=int,
        default=None,
        help="Force the recommended primary stream index in the sidecar instead of relying on automatic comparison.",
    )
    parser.add_argument(
        "--force-frame-format",
        type=str,
        default=None,
        help="Force the recommended frame format in the sidecar, for example equirect, fisheye, dfisheye, flat, c3x2, c6x1, c1x6.",
    )
    parser.add_argument(
        "--force-strategy",
        choices=sorted(RECOMMENDED_STRATEGIES),
        default=None,
        help="Force the recommended strategy in the sidecar instead of relying on automatic comparison.",
    )
    parser.add_argument("--clean", action="store_true", help="Delete previous preprocess sample folders before writing new ones.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing metadata and sample frames.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.sample_count < 1:
        raise ValueError("--sample-count must be at least 1")

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
        logger.info("Project folders checked and created if missing")

        ffmpeg_path = find_ffmpeg(code_root)
        ffprobe_path = find_ffprobe(code_root)
        logger.info("FFmpeg found: %s", ffmpeg_path)
        logger.info("ffprobe found: %s", ffprobe_path)

        input_video = resolve_input_video(args.input, paths["input_video"])
        if not input_video.exists():
            raise FileNotFoundError(f"Input video not found: {input_video}")

        forced_frame_format = normalize_frame_format(args.force_frame_format)
        sample_positions = parse_fraction_list(args.sample_positions)
        sample_fractions = build_sample_fractions(args.sample_count, sample_positions)

        if args.clean and paths["preprocess_samples"].exists():
            shutil.rmtree(paths["preprocess_samples"])
            logger.info("Removed existing preprocess samples: %s", paths["preprocess_samples"])
            paths["preprocess_samples"].mkdir(parents=True, exist_ok=True)

        ffprobe_payload = ffprobe_media_info(ffprobe_path, input_video, logger)
        format_payload = ffprobe_payload.get("format") or {}
        container_duration = float(format_payload.get("duration") or 0.0)

        all_video_streams, candidate_streams = summarize_streams(ffprobe_payload, logger)
        logger.info("Input video: %s", input_video)
        logger.info("Container duration (s): %.3f", container_duration)
        logger.info("Sample fractions: %s", sample_fractions)

        sample_timestamps = [timestamp_seconds(container_duration, fraction) for fraction in sample_fractions]

        stream_samples: list[dict] = []
        for stream in candidate_streams:
            stream_index = stream["stream_index"]
            sample_dir = paths["preprocess_samples"] / f"stream_{stream_index:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            stream_sample_paths: list[Path] = []
            stream_failed = False
            extracted_samples: list[dict] = []
            for timestamp_sec in sample_timestamps:
                sample_name = f"{serialize_timestamp_label(timestamp_sec)}.jpg"
                sample_path = sample_dir / sample_name

                try:
                    if sample_path.exists() and not args.overwrite:
                        logger.info("Reusing existing sample frame: %s", sample_path)
                    else:
                        extract_sample_frame(
                            ffmpeg_path=ffmpeg_path,
                            input_video=input_video,
                            stream_index=stream_index,
                            timestamp_sec=timestamp_sec,
                            output_path=sample_path,
                            logger=logger,
                        )
                    stream_sample_paths.append(sample_path)
                except Exception as exc:
                    logger.warning(
                        "Skipping stream %s because sample extraction failed at %.3fs: %s",
                        stream_index,
                        timestamp_sec,
                        exc,
                    )
                    stream_failed = True
                    break

                extracted_samples.append(
                    {
                        "timestamp_sec": round(timestamp_sec, 6),
                        "fraction": round(timestamp_sec / container_duration, 6) if container_duration > 0 else None,
                        "path": str(sample_path),
                        "name": sample_path.name,
                    }
                )

            if stream_failed:
                continue

            format_guess = infer_frame_format_from_image(Path(extracted_samples[0]["path"]))
            effective_frame_format = forced_frame_format or format_guess["format"]
            frame_format_source = "forced" if forced_frame_format else "auto"

            stream_samples.append(
                {
                    **stream,
                    "samples": extracted_samples,
                    "sample_dir": str(sample_dir),
                    "frame_format_guess": format_guess,
                    "effective_frame_format": effective_frame_format,
                    "effective_frame_format_source": frame_format_source,
                }
            )

            logger.info(
                "Stream %s guessed frame format: %s (effective=%s source=%s confidence=%.2f)",
                stream_index,
                format_guess["format"],
                effective_frame_format,
                frame_format_source,
                format_guess["confidence"],
            )

        pairwise_results: list[dict] = []
        for idx_a in range(len(stream_samples)):
            for idx_b in range(idx_a + 1, len(stream_samples)):
                stream_a = stream_samples[idx_a]
                stream_b = stream_samples[idx_b]

                sample_pairs = zip(stream_a["samples"], stream_b["samples"])
                frame_pair_metrics = [
                    compare_frame_pair(Path(sample_a["path"]), Path(sample_b["path"]))
                    for sample_a, sample_b in sample_pairs
                ]
                aggregate_metrics = aggregate_pairwise_metrics(frame_pair_metrics)
                classification, confidence, reasons = classify_stream_pair(aggregate_metrics)

                pairwise_results.append(
                    {
                        "stream_a": stream_a["stream_index"],
                        "stream_b": stream_b["stream_index"],
                        "classification": classification,
                        "confidence": confidence,
                        "reasons": reasons,
                        "aggregate_metrics": aggregate_metrics,
                        "frame_pair_metrics": frame_pair_metrics,
                    }
                )

                logger.info(
                    "Pairwise comparison stream %s vs %s -> %s (confidence %.2f)",
                    stream_a["stream_index"],
                    stream_b["stream_index"],
                    classification,
                    confidence,
                )

        recommendation = build_strategy_recommendation(
            candidate_streams=stream_samples,
            pairwise_results=pairwise_results,
            forced_primary_stream_index=args.force_primary_stream_index,
            forced_frame_format=forced_frame_format,
            forced_strategy=args.force_strategy,
        )

        if recommendation["recommended_frame_format"] is None:
            primary_idx = recommendation.get("recommended_primary_stream_index")
            if primary_idx is not None:
                primary_stream = next((stream for stream in stream_samples if stream["stream_index"] == primary_idx), None)
                if primary_stream is not None:
                    recommendation["recommended_frame_format"] = primary_stream["effective_frame_format"]
            elif len(stream_samples) == 1:
                recommendation["recommended_frame_format"] = stream_samples[0]["effective_frame_format"]

        preprocess_payload = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_script": SCRIPT_NAME,
            "mode": args.mode,
            "input_video": str(input_video),
            "input_video_name": input_video.name,
            "container": {
                "duration_sec": container_duration,
                "size_bytes": int(format_payload["size"]) if format_payload.get("size") not in (None, "N/A") else None,
                "bit_rate": int(format_payload["bit_rate"]) if format_payload.get("bit_rate") not in (None, "N/A") else None,
            },
            "sample_fractions": sample_fractions,
            "sample_timestamps_sec": [round(value, 6) for value in sample_timestamps],
            "force_overrides": {
                "primary_stream_index": args.force_primary_stream_index,
                "frame_format": forced_frame_format,
                "strategy": args.force_strategy,
            },
            "video_streams_all": all_video_streams,
            "candidate_streams": stream_samples,
            "pairwise_results": pairwise_results,
            "recommendation": recommendation,
            "tools": {
                "ffmpeg_path": str(ffmpeg_path),
                "ffprobe_path": str(ffprobe_path),
            },
            "notes": [
                "This is a report-first preprocess step. It recommends a strategy but does not yet perform stream stitching.",
                "Use --force-primary-stream-index, --force-frame-format, or --force-strategy when you want explicit manual control instead of the recommendation.",
            ],
        }

        paths["preprocess_metadata"].write_text(
            json.dumps(preprocess_payload, indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote preprocess metadata: %s", paths["preprocess_metadata"])
        logger.info("Recommended strategy: %s", recommendation["recommended_strategy"])
        logger.info("Recommended primary stream index: %s", recommendation["recommended_primary_stream_index"])
        logger.info("Recommended frame format: %s", recommendation["recommended_frame_format"])

        print("\n=== Preprocess Input Video Report ===")
        print(f"Input video: {input_video.name}")
        print(f"Mode: {args.mode}")
        print(f"Candidate video streams: {len(stream_samples)}")
        for stream in stream_samples:
            print(
                f"  - stream {stream['stream_index']}: "
                f"{stream['width']}x{stream['height']} codec={stream['codec_name']} "
                f"frame_format_guess={stream['frame_format_guess']['format']} "
                f"effective_frame_format={stream['effective_frame_format']}"
            )
        if pairwise_results:
            print("Pairwise similarity:")
            for pair in pairwise_results:
                metrics = pair["aggregate_metrics"]
                print(
                    f"  - {pair['stream_a']} vs {pair['stream_b']}: "
                    f"class={pair['classification']} confidence={pair['confidence']:.2f} "
                    f"mae={metrics['mean_absolute_difference']:.4f} "
                    f"corr={metrics['correlation']:.4f} "
                    f"dhash={metrics['dhash_hamming_distance']:.2f}"
                )
        print("Recommendation:")
        print(f"  strategy={recommendation['recommended_strategy']}")
        print(f"  primary_stream_index={recommendation['recommended_primary_stream_index']}")
        print(f"  frame_format={recommendation['recommended_frame_format']}")
        print(f"  metadata={paths['preprocess_metadata']}")

        logger.info("Preprocess input video completed successfully")
        return 0

    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
