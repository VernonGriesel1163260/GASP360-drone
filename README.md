# Gaussian Splats Preprocessing Pipeline

This repository provides a Windows-friendly preprocessing pipeline for turning a **360 video** into a **COLMAP sparse reconstruction** suitable for Gaussian Splatting and related workflows.

## What It Does

The pipeline runs these stages:

1. Extract frames from a 360 video (`scripts/extract_frames.py`)
2. Convert equirectangular frames to perspective views (`scripts/convert_360_to_views.py`)
3. Flatten view images into a COLMAP input folder (`scripts/prepare_colmap_images.py`)
4. Run sparse reconstruction (`scripts/run_colmap.py`)

There is also an orchestrator script (`scripts/pipeline.py`) to run all stages end-to-end.

## Repository Layout

```text
scripts/
  pipeline.py
  extract_frames.py
  convert_360_to_views.py
  prepare_colmap_images.py
  run_colmap.py
  common/
data/
  input_video/
  frames_360/
  frames_perspective/
  colmap/
logs/
COLMAP/
tools/
```

## Requirements

- Windows (PowerShell commands below are Windows style)
- Python 3.10+
- FFmpeg (`ffmpeg.exe`, optionally `ffprobe.exe`)
- COLMAP installation (local `COLMAP` folder is supported)

Python packages:

```powershell
python -m pip install -r requirements.txt
```

## Tool Discovery

### FFmpeg

`extract_frames.py` and `convert_360_to_views.py` search in this order:

1. `tools/ffmpeg/bin/ffmpeg.exe`
2. `tools/ffmpeg.exe`
3. `ffmpeg` on `PATH`

`extract_frames.py` uses `ffprobe` for `--target-frames` mode.

### COLMAP Backend Priority

`run_colmap.py` selects backend in this order:

1. `pycolmap` (unless `--force-cli`)
2. `COLMAP/bin/colmap.exe`
3. `COLMAP/COLMAP.bat`

If no backend is found, it exits with a clear error.

## Quick Start

Place your 360 source video in `data/input_video/`.

Run full pipeline:

```powershell
python .\scripts\pipeline.py --preset indoor_real_estate --overwrite --clean-convert --clean-prepare --reset-colmap --verbose
```

Dry run (show commands only):

```powershell
python .\scripts\pipeline.py --dry-run --verbose
```

## Run Steps Manually

### 1) Extract frames

```powershell
python .\scripts\extract_frames.py --target-frames 100 --overwrite --clean --verbose
```

Or explicit input:

```powershell
python .\scripts\extract_frames.py --input .\data\input_video\house360.mp4 --target-frames 150 --overwrite --clean --verbose
```

### 2) Convert 360 frames to perspective views

```powershell
python .\scripts\convert_360_to_views.py --preset indoor_real_estate --input-prefix frame360 --overwrite --clean --verbose
```

Quick test:

```powershell
python .\scripts\convert_360_to_views.py --preset indoor_real_estate --input-prefix frame360 --limit 5 --overwrite --clean --verbose
```

### 3) Prepare COLMAP images

```powershell
python .\scripts\prepare_colmap_images.py --input-prefix frame360 --copy-mode copy --clean --verbose
```

Strict mode (require complete view sets):

```powershell
python .\scripts\prepare_colmap_images.py --input-prefix frame360 --strict --clean --verbose
```

### 4) Run COLMAP sparse reconstruction

```powershell
python .\scripts\run_colmap.py --preset indoor_real_estate --reset --verbose
```

Force CLI backend:

```powershell
python .\scripts\run_colmap.py --preset indoor_real_estate --force-cli --reset --verbose
```

## Using Pipeline to run end-to-end processing
How to use it

Full run with your default indoor preset:

python .\scripts\pipeline.py --preset indoor_real_estate --input-video ".\data\input_video\house360.mp4" --overwrite --clean-extract --clean-convert --clean-prepare --reset-colmap --force-cli --verbose

Run from conversion onward, skipping extraction:

python .\scripts\pipeline.py --preset indoor_real_estate --step-from convert_360_to_views --overwrite --clean-convert --clean-prepare --reset-colmap --force-cli --verbose

Run only through COLMAP prep, not reconstruction:

python .\scripts\pipeline.py --preset tight_interiors --input-video ".\data\input_video\house360.mp4" --step-to prepare_colmap_images --overwrite --clean-extract --clean-convert --clean-prepare --verbose

Dry run to inspect the generated commands:

python .\scripts\pipeline.py --preset corridor_staircase --input-video ".\data\input_video\house360.mp4" --dry-run
Recommended first production-style command

For your current real-estate workflow:

python .\scripts\pipeline.py --preset indoor_real_estate --input-video ".\data\input_video\house360.

## Presets

Presets are defined in `scripts/common/presets.py` and control projection + COLMAP defaults.

Available presets:

- `indoor_real_estate`
- `outdoor_drone`
- `tight_interiors`
- `corridor_staircase`
- `mixed_property_tour`
- `custom`

Use preset with optional overrides:

```powershell
python .\scripts\convert_360_to_views.py --preset tight_interiors --input-prefix frame360 --h-fov 78 --v-fov 78 --overwrite
python .\scripts\run_colmap.py --preset outdoor_drone --camera-model OPENCV --matcher sequential_matcher --force-cli --reset --verbose
```

## Logs

Each script writes logs under `logs/<script_name>/` and also updates a `*_latest.log` file for quick access.

## Common Issues

### No input video found

Put a video in `data/input_video/` or pass `--input` to `extract_frames.py`.

### `--target-frames` fails

`ffprobe` is required for duration-based FPS calculation.

### COLMAP option errors on CLI backend

`run_colmap.py` checks whether options are supported by your installed COLMAP before appending optional flags.

## Notes

- `run_glomap.py` exists as a separate legacy/experimental utility and is not part of the main scripted pipeline above.

## TODO: 
So the pipeline summary is almost certainly counting a line that merely contains the word error, rather than a real failing condition. That means:

--stop-on-warning is usable
but the error/warning classification is not trustworthy enough yet for strong automation decisions
