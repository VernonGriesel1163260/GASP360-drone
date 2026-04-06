from __future__ import annotations

from copy import deepcopy
from typing import Any


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value

    return result


def build_projection_override_dict(args) -> dict[str, Any]:
    override: dict[str, Any] = {}

    if getattr(args, "views", None):
        override["views"] = args.views

    for field in ("h_fov", "v_fov", "width", "height", "pitch", "roll", "interpolation", "quality"):
        value = getattr(args, field, None)
        if value is not None:
            override[field] = value

    return override


def build_colmap_override_dict(args) -> dict[str, Any]:
    override: dict[str, Any] = {}

    for field in ("matcher", "camera_model", "max_image_size"):
        value = getattr(args, field, None)
        if value is not None:
            override[field] = value

    if getattr(args, "single_camera", False):
        override["single_camera"] = True

    if getattr(args, "use_gpu", False):
        override["use_gpu"] = True

    return override