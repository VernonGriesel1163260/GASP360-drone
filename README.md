# Gaussian Splats Preprocessing Pipeline

A Windows-friendly preprocessing pipeline for turning 360 or wide-angle video into a COLMAP sparse reconstruction suitable for Gaussian Splatting and related workflows.

## Overview

The pipeline runs these stages:

1. **Extract frames** from a source video (`scripts/extract_frames.py`)
2. **Convert source frames** into perspective views (`scripts/convert_360_to_views.py`)
3. **Flatten view images** into a COLMAP input folder (`scripts/prepare_colmap_images.py`)
4. **Run COLMAP** sparse reconstruction (`scripts/run_colmap.py`)

There is also an orchestrator script, `scripts/pipeline.py`, to run the stages end to end.

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
tools/
COLMAP/
```

## Requirements

- Windows
- Python 3.10+
- FFmpeg (`ffmpeg.exe`, optionally `ffprobe.exe`)
- COLMAP installation or `pycolmap`

Install Python dependencies:

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

### COLMAP

`run_colmap.py` selects a backend in this order:

1. `pycolmap` (unless `--force-cli`)
2. `COLMAP/bin/colmap.exe`
3. `COLMAP/COLMAP.bat`

If no backend is found, the script exits with a clear error.

## Supported Input Video Containers

The extractor now accepts multiple source video/container extensions, including:

- `.osv`
- `.insv`
- `.360`
- `.mp4`
- `.mkv`
- `.mov`
- `.avi`
- `.webm`
- `.mts`
- `.m2ts`

Put your source video in `data/input_video/`, or pass a full explicit path with `--input-video`.

## Quick Start

### Full pipeline with automatic input discovery

```powershell
python .\scripts\pipeline.py `
  --preset indoor_real_estate `
  --overwrite `
  --clean-extract `
  --clean-convert `
  --clean-prepare `
  --reset-colmap `
  --verbose
```

### Full pipeline with explicit input file

```powershell
python .\scripts\pipeline.py `
  --preset indoor_real_estate `
  --input-video ".\data\input_video\DJI_20260328162848_0011_D.OSV" `
  --overwrite `
  --clean-extract `
  --clean-convert `
  --clean-prepare `
  --reset-colmap `
  --verbose
```

### Dry run

```powershell
python .\scripts\pipeline.py --dry-run --verbose
```

## Input Projection Formats

The conversion stage now supports different **input frame layouts** through `--input-format`.

This matters because FFmpeg's `v360` filter needs to know how the incoming frame is encoded before it can correctly remap it into perspective views.

### Quick rule of thumb

- **Wide 2:1 panorama strip** → `equirect`
- **One circular lens image** → `fisheye`
- **Two circular lens images in one frame** → `dfisheye`
- **Normal camera footage** → `flat`
- **Cube-face layouts** → `c3x2`, `c6x1`, `c1x6`, or `eac`

### Supported aliases in this repo

| Friendly value | Canonical format passed to FFmpeg |
|---|---|
| `equirect`, `equirectangular`, `e` | `equirect` |
| `flat`, `plain`, `rectilinear`, `gnomonic` | `flat` |
| `fisheye`, `fish-eye`, `fish_eye`, `fishere`, `fishereye` | `fisheye` |
| `dual-fisheye`, `dfisheye`, `dualfisheye` | `dfisheye` |
| `cubemap`, `cubemap-3x2`, `c3x2` | `c3x2` |
| `cubemap-6x1`, `c6x1` | `c6x1` |
| `cubemap-1x6`, `c1x6` | `c1x6` |
| `eac`, `equiangular-cubemap` | `eac` |
| `stereographic`, `little-planet`, `sg` | `sg` |
| `half-equirect`, `hequirect`, `he` | `he` |
| `orthographic`, `og` | `og` |
| plus advanced formats | `mercator`, `ball`, `hammer`, `sinusoidal`, `pannini`, `cylindrical`, `tetrahedron`, `tsp`, `equisolid`, `octahedron`, `cylindricalea`, `barrel`, `fb`, `barrelsplit` |

> `perspective` is output-only in FFmpeg v360, so it is **not valid** as an `--input-format`.

### Recommended defaults

For most practical reconstruction work, start with one of these:

| Situation | Suggested `--input-format` | Notes |
|---|---|---|
| Stitched 360 export | `equirect` | Best default for most consumer 360 exports |
| Raw single circular ultra-wide frame | `fisheye` | Usually pair with `--input-h-fov 180 --input-v-fov 180` |
| Raw dual-lens frame | `dfisheye` | Often pair with `--input-h-fov 180 --input-v-fov 180` |
| Normal non-360 footage | `flat` | Usually safest with `--views front` only |
| Existing cube-face output | `c3x2`, `c6x1`, `c1x6`, or `eac` | Use the layout that matches the actual frame packing |

## Input Format Gallery

### Common formats

#### `equirect`

![equirect](docs/input_formats/input-format-equirect.png)

**What it looks like:** a single wide 2:1 panorama strip.  
**When to use it:** use this for most already-stitched 360 exports.

#### `flat`

![flat](docs/input_formats/input-format-flat.png)

**What it looks like:** a normal camera frame.  
**When to use it:** use this for plain footage. Usually safest with `--views front` only.

#### `fisheye`

![fisheye](docs/input_formats/input-format-fisheye.png)

**What it looks like:** one circular lens image with strong edge curvature.  
**When to use it:** use this when a single fisheye lens fills the frame. Start with `--input-h-fov 180 --input-v-fov 180`.

#### `dfisheye`

![dfisheye](docs/input_formats/input-format-dfisheye.png)

**What it looks like:** two circular fisheye views in one frame.  
**When to use it:** use this for raw dual-lens 360 camera outputs before stitching.

#### `c3x2`

![c3x2](docs/input_formats/input-format-c3x2.png)

**What it looks like:** six square faces in a 3×2 grid.  
**When to use it:** use this when the source is already packed as a cubemap.

#### `eac`

![eac](docs/input_formats/input-format-eac.png)

**What it looks like:** cube faces with more even angular sampling.  
**When to use it:** use this when the source was exported as an Equi-Angular Cubemap.

### Additional supported formats

#### `c6x1`

![c6x1](docs/input_formats/input-format-c6x1.png)

**What it looks like:** cubemap faces in one horizontal strip.  
**When to use it:** same content as other cubemaps, just a different packing layout.

#### `c1x6`

![c1x6](docs/input_formats/input-format-c1x6.png)

**What it looks like:** cubemap faces in one vertical strip.  
**When to use it:** same as other cubemap layouts, but packed vertically.

#### `barrel` / `fb` / `barrelsplit`

![barrel](docs/input_formats/input-format-barrel.png)

**What it looks like:** split barrel-style Facebook 360 layouts.  
**When to use it:** only when your source clearly matches this legacy encoding.

#### `sg`

![sg](docs/input_formats/input-format-sg.png)

**What it looks like:** a circular little-planet style projection.  
**When to use it:** use this only when the source already looks stereographic.

#### `mercator`

![mercator](docs/input_formats/input-format-mercator.png)

**What it looks like:** a map-like projection with vertical stretching near top and bottom.  
**When to use it:** mainly for conversion workflows, not camera-native input.

#### `ball`

![ball](docs/input_formats/input-format-ball.png)

**What it looks like:** a sphere-like projection with severe distortion toward the back.  
**When to use it:** niche conversion format.

#### `hammer`

![hammer](docs/input_formats/input-format-hammer.png)

**What it looks like:** an oval world-map style layout.  
**When to use it:** niche panorama conversion format.

#### `sinusoidal`

![sinusoidal](docs/input_formats/input-format-sinusoidal.png)

**What it looks like:** a curved-edge map projection.  
**When to use it:** niche panorama conversion format.

#### `pannini`

![pannini](docs/input_formats/input-format-pannini.png)

**What it looks like:** a wide-angle projection that preserves verticals better than plain perspective.  
**When to use it:** useful for some remap workflows, not usually camera-native.

#### `cylindrical`

![cylindrical](docs/input_formats/input-format-cylindrical.png)

**What it looks like:** a horizontal wrap around a cylinder.  
**When to use it:** mostly an intermediate projection.

#### `he`

![he](docs/input_formats/input-format-he.png)

**What it looks like:** a half-height panorama strip.  
**When to use it:** when the source covers only part of the sphere, often a hemisphere.

#### `equisolid`

![equisolid](docs/input_formats/input-format-equisolid.png)

**What it looks like:** a fisheye-family projection with a different angle-to-radius mapping.  
**When to use it:** only when you know your lens model is equisolid.

#### `og`

![og](docs/input_formats/input-format-og.png)

**What it looks like:** a globe-like orthographic projection.  
**When to use it:** niche conversion format.

#### `tetrahedron`

![tetrahedron](docs/input_formats/input-format-tetrahedron.png)

**What it looks like:** four triangular faces.  
**When to use it:** rare, but useful if another tool already exported this packing.

#### `tsp`

![tsp](docs/input_formats/input-format-tsp.png)

**What it looks like:** a truncated square pyramid style packing.  
**When to use it:** niche spherical workflow.

#### `octahedron`

![octahedron](docs/input_formats/input-format-octahedron.png)

**What it looks like:** eight triangular faces.  
**When to use it:** another polygonal sphere packing format.

#### `cylindricalea`

![cylindricalea](docs/input_formats/input-format-cylindricalea.png)

**What it looks like:** cylindrical equal-area map projection.  
**When to use it:** mainly for conversion or analysis workflows.

### Full cheat sheet

![Input format cheat sheet](docs/input-format-cheatsheet.png)

## How to Choose the Right Input Format

### Start here

1. Open a frame from `data/frames_360/`
2. Compare its shape against the gallery above
3. Pick the matching `--input-format`
4. If the layout is lens-based, set input FOV values too
5. Run a small test with `--limit 3` before doing a full conversion

### Practical examples

#### Stitched 360 export

```powershell
python .\scripts\pipeline.py `
  --preset indoor_real_estate `
  --input-video ".\data\input_video\DJI_20260328162848_0011_D.OSV" `
  --input-format equirect `
  --overwrite `
  --clean-extract `
  --clean-convert `
  --clean-prepare `
  --reset-colmap `
  --verbose
```

#### Single fisheye source

```powershell
python .\scripts\pipeline.py `
  --preset indoor_real_estate `
  --input-video ".\data\input_video\your_fisheye.mp4" `
  --input-format fisheye `
  --input-h-fov 180 `
  --input-v-fov 180 `
  --overwrite `
  --clean-extract `
  --clean-convert `
  --clean-prepare `
  --reset-colmap `
  --verbose
```

#### Dual-fisheye source

```powershell
python .\scripts\pipeline.py `
  --preset indoor_real_estate `
  --input-video ".\data\input_video\your_dual_fisheye.mp4" `
  --input-format dfisheye `
  --input-h-fov 180 `
  --input-v-fov 180 `
  --overwrite `
  --clean-extract `
  --clean-convert `
  --clean-prepare `
  --reset-colmap `
  --verbose
```

#### Plain flat footage

```powershell
python .\scripts\pipeline.py `
  --preset indoor_real_estate `
  --input-video ".\data\input_video\walkthrough.mp4" `
  --input-format flat `
  --views front `
  --overwrite `
  --clean-extract `
  --clean-convert `
  --clean-prepare `
  --reset-colmap `
  --verbose
```

### Quick conversion-only smoke test

```powershell
python .\scripts\convert_360_to_views.py `
  --preset indoor_real_estate `
  --input-format equirect `
  --limit 3 `
  --overwrite `
  --clean `
  --verbose
```

## Run Steps Manually

### 1) Extract frames

```powershell
python .\scripts\extract_frames.py --target-frames 100 --overwrite --clean --verbose
```

Or explicit input:

```powershell
python .\scripts\extract_frames.py `
  --input ".\data\input_video\DJI_20260328162848_0011_D.OSV" `
  --target-frames 150 `
  --overwrite `
  --clean `
  --verbose
```

### 2) Convert frames to perspective views

```powershell
python .\scripts\convert_360_to_views.py `
  --preset indoor_real_estate `
  --input-prefix frame360 `
  --input-format equirect `
  --overwrite `
  --clean `
  --verbose
```

### 3) Prepare COLMAP images

```powershell
python .\scripts\prepare_colmap_images.py `
  --input-prefix frame360 `
  --copy-mode copy `
  --clean `
  --verbose
```

### 4) Run COLMAP sparse reconstruction

```powershell
python .\scripts\run_colmap.py --preset indoor_real_estate --reset --verbose
```

## Presets

Presets are defined in `scripts/common/presets.py` and control projection plus COLMAP defaults.

Available presets:

- `indoor_real_estate`
- `outdoor_drone`
- `tight_interiors`
- `corridor_staircase`
- `mixed_property_tour`
- `custom`

Examples:

```powershell
python .\scripts\convert_360_to_views.py `
  --preset tight_interiors `
  --input-prefix frame360 `
  --input-format equirect `
  --h-fov 78 `
  --v-fov 78 `
  --overwrite

python .\scripts\run_colmap.py `
  --preset outdoor_drone `
  --camera-model OPENCV `
  --matcher sequential_matcher `
  --force-cli `
  --reset `
  --verbose
```

## Experiments

Run all experiments:

```powershell
python .\scripts\run_experiments.py --config .\experiments.yaml --verbose
```

Resume later:

```powershell
python .\scripts\run_experiments.py --config .\experiments.yaml --resume --verbose
```

Run one experiment from conversion onward:

```powershell
python .\scripts\run_experiments.py `
  --config .\experiments.yaml `
  --experiment indoor_h95 `
  --step-from convert_360_to_views `
  --verbose
```

## Visualising Experiment Results

After a matrix run, expect summary files like:

```text
experiments\experiments_summary.csv
experiments\experiments_summary.json
```

Generate plots and a markdown summary:

```powershell
python .\scripts\visualize_experiments.py `
  --summary-csv ".\experiments\experiments_summary.csv" `
  --output-dir ".\experiments\_reports"
```

## Brush

Prepare dataset only:

```powershell
python .\scripts\run_brush.py --prepare-only --clean-input --verbose
```

Run Brush on current workspace:

```powershell
python .\scripts\run_brush.py --clean-input --with-viewer --verbose
```

Target a specific experiment workspace:

```powershell
$env:GASP_WORKSPACE_ROOT="E:\_root\projects\Gaussian Splats\GASP360-drone\experiments\indoor_h95"
python .\scripts\run_brush.py --clean-input --verbose
```

## Logs

Each script writes logs under `logs/<script_name>/` and updates a `*_latest.log` file for quick access.

## Common Issues

### Input video not found

Check the real filename in `data/input_video/`.

```powershell
Get-ChildItem .\data\input_video
```

Then pass the exact filename:

```powershell
python .\scripts\pipeline.py --input-video ".\data\input_video\DJI_20260328162848_0011_D.OSV"
```

### Unsure which `--input-format` to use

Open a sample frame and compare it to the gallery in this README. Start with:

- `equirect` for stitched 360 panoramas
- `fisheye` for one circular lens image
- `dfisheye` for two circular lens images
- `flat` for ordinary footage

### `--target-frames` fails

`ffprobe` is required for duration-based FPS calculation.

### COLMAP CLI option errors

`run_colmap.py` checks whether optional CLI flags are supported by your installed COLMAP before appending them.

## Notes

- `run_glomap.py` exists as a legacy or experimental utility and is not part of the main pipeline above.
- For real reconstruction work, the most important practical choices are usually: source layout, FOV values, number of generated views, and COLMAP matcher settings.


## TODO: Add later in README:

### What this auto-detect is good at

```powershell
It should do reasonably well for the common cases:

equirect
fisheye
dfisheye
c3x2
c6x1
c1x6
fallback flat

It is not highly reliable for more exotic layouts like:

eac
barrel
sg
mercator
pannini
custom stitched variants

Those should still be set manually.
```


