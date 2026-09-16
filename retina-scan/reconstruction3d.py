"""Safe relative optic-disc topography from a single colour fundus image.

This cannot reconstruct physical depth: a single monocular fundus photograph
has no depth calibration. OCT or stereo imaging is needed for clinical cup
depth. This module instead draws a repeatable *relative* cup-and-rim shape
from the segmented disc and cup, for explaining the VCDR measurement only.
"""
from __future__ import annotations

import cv2
import numpy as np
import plotly.graph_objects as go

import multi_disease

_SIZE = 512
_MIN_DISC_RADIUS = 26
_MIN_DISC_AREA = 1800
_MIN_CUP_AREA = 80


def _normalise_rgb(img_rgb: np.ndarray) -> np.ndarray:
    img = np.asarray(img_rgb)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("expected an HxWx3 RGB fundus image")
    if min(img.shape[:2]) < 96:
        raise ValueError("image is too small to assess the optic disc")
    return cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (_SIZE, _SIZE),
                      interpolation=cv2.INTER_AREA)


def _disc_geometry(img: np.ndarray) -> dict:
    """Locate a complete disc, then use the shared cup segmentation."""
    green = img[:, :, 1]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fov = multi_disease._fov_mask(gray)
    if int(np.count_nonzero(fov)) < _SIZE * _SIZE * 0.15:
        raise ValueError("retinal field of view is too small")
    center, radius = multi_disease._locate_disc(green, fov)
    cx, cy = center
    if radius < _MIN_DISC_RADIUS:
        raise ValueError("optic disc is too small or could not be located")
    margin = int(radius * 1.25)
    if cx - margin < 0 or cy - margin < 0 or cx + margin >= _SIZE or cy + margin >= _SIZE:
        raise ValueError("optic disc is too close to the image edge")
    if fov[cy, cx] == 0:
        raise ValueError("optic disc lies outside the retinal field of view")

    cdr = multi_disease._cup_to_disc(green, center, radius)
    cup_local = cdr["cup_mask"].astype(np.uint8)
    x0, y0 = cdr["patch_origin"]
    cup = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
    h, w = cup_local.shape
    cup[y0:y0 + h, x0:x0 + w] = cup_local
    disc = np.zeros((_SIZE, _SIZE), dtype=np.uint8)
    cv2.circle(disc, center, radius, 1, -1)
    cup &= disc
    cup_area, disc_area = int(np.count_nonzero(cup)), int(np.count_nonzero(disc))
    if disc_area < _MIN_DISC_AREA or cup_area < _MIN_CUP_AREA:
        raise ValueError("cup/disc segmentation is not reliable enough")

    moments = cv2.moments(cup)
    if moments["m00"] == 0:
        raise ValueError("cup segmentation is empty")
    cup_center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    if np.hypot(cup_center[0] - cx, cup_center[1] - cy) > radius * 0.45:
        raise ValueError("segmented cup is not centred in the optic disc")
    local = green[cy - radius:cy + radius, cx - radius:cx + radius]
    if float(local.std()) < 7.0:
        raise ValueError("optic-disc contrast is too low")
    return {"center": center, "radius": radius, "disc": disc, "cup": cup,
            "vcdr": cdr["vcdr"], "disc_area": disc_area, "cup_area": cup_area}


def _relative_surface(geometry: dict) -> np.ndarray:
    """Mask-derived, unitless bowl; image brightness never becomes depth."""
    disc = geometry["disc"].astype(bool)
    cup = geometry["cup"].astype(np.uint8)
    cx, cy = geometry["center"]
    yy, xx = np.indices(disc.shape)
    radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(geometry["radius"], 1)
    z = np.zeros(disc.shape, dtype=np.float32)
    z[disc] = 0.10 * np.clip(1.0 - radial[disc] ** 2, 0.0, 1.0)
    cup_distance = cv2.distanceTransform(cup, cv2.DIST_L2, 5)
    if cup_distance.max() <= 0:
        raise ValueError("cup shape could not be transformed into a surface")
    bowl = cup_distance / cup_distance.max()
    display_depth = 0.20 + 0.25 * float(np.clip(geometry["vcdr"], 0.0, 0.95))
    z[cup > 0] -= display_depth * bowl[cup > 0]
    return cv2.GaussianBlur(z, (0, 0), 1.1) * disc


def analyze_optic_disc(img_rgb: np.ndarray) -> dict:
    """Return a figure or a clear reason why no safe figure is available."""
    try:
        img = _normalise_rgb(img_rgb)
        geometry = _disc_geometry(img)
        z = _relative_surface(geometry)
        surface = np.where(geometry["disc"].astype(bool), z, np.nan)
        fig = go.Figure(data=[go.Surface(
            z=surface, surfacecolor=surface, colorscale="RdYlBu_r",
            cmin=-0.5, cmax=0.15, showscale=False, hoverinfo="skip",
            contours={"z": {"show": True, "usecolormap": True, "highlight": False}},
        )])
        fig.update_layout(
            title="Relative optic-disc topography (explanatory only)",
            autosize=True, margin=dict(l=0, r=0, b=0, t=36),
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                       zaxis=dict(visible=False), aspectmode="manual",
                       aspectratio=dict(x=1, y=1, z=0.42),
                       camera=dict(eye=dict(x=1.35, y=-1.35, z=0.9))),
        )
        return {
            "available": True, "figure": fig, "reason": "", "vcdr": geometry["vcdr"],
            "disc_diameter_px": geometry["radius"] * 2,
            "cup_area_ratio": round(geometry["cup_area"] / geometry["disc_area"], 2),
            "note": "Relative shape from segmented disc and cup; not physical depth or an OCT measurement.",
        }
    except Exception as exc:
        return {"available": False, "figure": None, "reason": str(exc),
                "note": "Relative topography is unavailable for this capture."}


def extract_disc_patch(img_rgb: np.ndarray):
    """Compatibility helper. New callers should use ``analyze_optic_disc``."""
    img = _normalise_rgb(img_rgb)
    g = _disc_geometry(img)
    cx, cy = g["center"]
    pad = int(g["radius"] * 1.25)
    return img[cy - pad:cy + pad, cx - pad:cx + pad, 1], img[cy - pad:cy + pad, cx - pad:cx + pad]


def build_3d_optic_disc(green_patch: np.ndarray, color_patch: np.ndarray | None = None):
    """Deprecated compatibility wrapper; validates the supplied colour patch."""
    return None if color_patch is None else analyze_optic_disc(color_patch).get("figure")
