#!/usr/bin/env python3
"""Helpers for row-id parsing and projected point extraction from geojson maps."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator

from pyproj import Transformer


ROW_ID_RE = re.compile(r"(ROW\d+)", re.IGNORECASE)


def _as_clean_string(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text if text else ""


def extract_row_id(props: Dict[str, Any] | None) -> str:
    """Extract row id from a feature properties dict.

    Priority order:
    1) ``vine_vine_row_id``
    2) ``ROW\\d+`` regex from ``row_post_id``
    3) ``ROW\\d+`` regex from ``feature_name``
    4) legacy ``_post_`` prefix from ``row_post_id`` or ``feature_name``
    5) ``unknown``
    """
    if not props:
        return "unknown"

    vine_row_id = _as_clean_string(props.get("vine_vine_row_id"))
    if vine_row_id:
        return vine_row_id

    for key in ("row_post_id", "feature_name"):
        text = _as_clean_string(props.get(key))
        if not text:
            continue
        match = ROW_ID_RE.search(text)
        if match:
            return match.group(1).upper()

    for key in ("row_post_id", "feature_name"):
        text = _as_clean_string(props.get(key))
        if "_post_" in text:
            return text.rsplit("_post_", 1)[0]

    return "unknown"


def iter_projected_points(
    geojson_path: Path | str,
    target_crs: str = "epsg:32630",
    source_crs: str = "epsg:4326",
) -> Iterator[Dict[str, Any]]:
    """Yield projected Point features as dictionaries with common fields.

    Returned dict keys:
    - ``x``, ``y``: projected coordinates in target CRS
    - ``row_id``: parsed row id
    - ``feature_type``: lowercased feature type string (possibly empty)
    - ``properties``: original feature properties dict
    """
    path = Path(geojson_path)
    if not path.exists():
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates", [])
        if len(coords) < 2:
            continue
        lon, lat = coords[0], coords[1]
        x, y = transformer.transform(lon, lat)
        props = feat.get("properties") or {}
        yield {
            "x": float(x),
            "y": float(y),
            "row_id": extract_row_id(props),
            "feature_type": _as_clean_string(props.get("feature_type")).lower(),
            "properties": props,
        }
