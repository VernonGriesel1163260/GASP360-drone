from __future__ import annotations

SCENARIO_PRESETS = {
    "indoor_real_estate": {
        "projection": {
            "views": ["front", "right", "back", "left"],
            "view_yaws": {
                "front": 0,
                "right": 90,
                "back": 180,
                "left": -90,
            },
            "h_fov": 85.0,
            "v_fov": 85.0,
            "width": 1280,
            "height": 1280,
            "pitch": 0.0,
            "roll": 0.0,
            "interpolation": "lanczos",
            "quality": 2,
        },
        "colmap": {
            "matcher": "sequential_matcher",
            "camera_model": "SIMPLE_RADIAL",
            "single_camera": True,
            "use_gpu": False,
            "max_image_size": None,
        },
    },
    "outdoor_drone": {
        "projection": {
            "views": ["front", "right", "back", "left"],
            "view_yaws": {
                "front": 0,
                "right": 90,
                "back": 180,
                "left": -90,
            },
            "h_fov": 95.0,
            "v_fov": 95.0,
            "width": 1400,
            "height": 1400,
            "pitch": 0.0,
            "roll": 0.0,
            "interpolation": "lanczos",
            "quality": 2,
        },
        "colmap": {
            "matcher": "sequential_matcher",
            "camera_model": "SIMPLE_RADIAL",
            "single_camera": True,
            "use_gpu": False,
            "max_image_size": None,
        },
    },
    "tight_interiors": {
        "projection": {
            "views": ["front", "front_right", "right", "back", "left", "front_left"],
            "view_yaws": {
                "front": 0,
                "front_right": 60,
                "right": 120,
                "back": 180,
                "left": -120,
                "front_left": -60,
            },
            "h_fov": 75.0,
            "v_fov": 75.0,
            "width": 1280,
            "height": 1280,
            "pitch": 0.0,
            "roll": 0.0,
            "interpolation": "lanczos",
            "quality": 2,
        },
        "colmap": {
            "matcher": "sequential_matcher",
            "camera_model": "SIMPLE_RADIAL",
            "single_camera": True,
            "use_gpu": False,
            "max_image_size": None,
        },
    },
    "corridor_staircase": {
        "projection": {
            "views": ["v000", "v045", "v090", "v135", "v180", "v225", "v270", "v315"],
            "view_yaws": {
                "v000": 0,
                "v045": 45,
                "v090": 90,
                "v135": 135,
                "v180": 180,
                "v225": -135,
                "v270": -90,
                "v315": -45,
            },
            "h_fov": 80.0,
            "v_fov": 80.0,
            "width": 1280,
            "height": 1280,
            "pitch": 0.0,
            "roll": 0.0,
            "interpolation": "lanczos",
            "quality": 2,
        },
        "colmap": {
            "matcher": "sequential_matcher",
            "camera_model": "SIMPLE_RADIAL",
            "single_camera": True,
            "use_gpu": False,
            "max_image_size": None,
        },
    },
    "mixed_property_tour": {
        "projection": {
            "views": ["front", "front_right", "right", "back", "left", "front_left"],
            "view_yaws": {
                "front": 0,
                "front_right": 60,
                "right": 120,
                "back": 180,
                "left": -120,
                "front_left": -60,
            },
            "h_fov": 85.0,
            "v_fov": 85.0,
            "width": 1280,
            "height": 1280,
            "pitch": 0.0,
            "roll": 0.0,
            "interpolation": "lanczos",
            "quality": 2,
        },
        "colmap": {
            "matcher": "sequential_matcher",
            "camera_model": "SIMPLE_RADIAL",
            "single_camera": True,
            "use_gpu": False,
            "max_image_size": None,
        },
    },
    "custom": {
        "projection": {
            "views": ["front", "right", "back", "left"],
            "view_yaws": {
                "front": 0,
                "right": 90,
                "back": 180,
                "left": -90,
            },
            "h_fov": 85.0,
            "v_fov": 85.0,
            "width": 1280,
            "height": 1280,
            "pitch": 0.0,
            "roll": 0.0,
            "interpolation": "lanczos",
            "quality": 2,
        },
        "colmap": {
            "matcher": "sequential_matcher",
            "camera_model": "SIMPLE_RADIAL",
            "single_camera": True,
            "use_gpu": False,
            "max_image_size": None,
        },
    },
}

DEFAULT_PRESET_NAME = "indoor_real_estate"


def get_preset_names() -> list[str]:
    return list(SCENARIO_PRESETS.keys())


def get_preset(name: str) -> dict:
    if name not in SCENARIO_PRESETS:
        raise KeyError(f"Unknown preset: {name}")
    return SCENARIO_PRESETS[name]