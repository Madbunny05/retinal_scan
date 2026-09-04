"""
Fundus image preprocessing + quality assessment.

Pure NumPy / OpenCV so it stays light and importable without torch.
The pipeline mirrors the well-known APTOS / Ben-Graham approach that most
diabetic-retinopathy CNNs are trained on:

    load -> crop to the retina disc -> circular mask -> contrast enhance -> resize

`quality_check` runs first so we can warn the user about blurry / dark /
non-fundus captures before wasting a prediction on them.
"""
from __future__ import annotations

import io
from typing import Union

import cv2
import numpy as np
from PIL import Image

import config

ArrayLike = Union[np.ndarray, Image.Image, bytes, str]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_rgb(src: ArrayLike) -> np.ndarray:
    """Return an HxWx3 uint8 RGB array from many possible inputs."""
    if isinstance(src, np.ndarray):
        img = src
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return np.ascontiguousarray(img[:, :, :3].astype(np.uint8))

    if isinstance(src, Image.Image):
        return np.array(src.convert("RGB"), dtype=np.uint8)

    if isinstance(src, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(src)).convert("RGB")
        return np.array(pil, dtype=np.uint8)

    if isinstance(src, str):
        pil = Image.open(src).convert("RGB")
        return np.array(pil, dtype=np.uint8)

    raise TypeError(f"Unsupported image source: {type(src)}")


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def crop_to_fundus(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop the black border around the circular retina field of view."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() < 100:                       # essentially a black frame
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    cropped = img[r0:r1 + 1, c0:c1 + 1]
    return cropped if cropped.size else img


def circular_mask(img: np.ndarray, fill: int = 128) -> np.ndarray:
    """Blank out the corners outside the circular FOV to remove edge artefacts."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = int(min(h, w) / 2 * 0.96)
    cv2.circle(mask, center, radius, 255, -1)
    out = img.copy()
    out[mask == 0] = fill
    return out


# --------------------------------------------------------------------------
# Enhancement
# --------------------------------------------------------------------------
def ben_graham(img: np.ndarray, scale: float = 30.0) -> np.ndarray:
    """Subtract a local average colour to boost micro-lesion contrast."""
    sigma = max(img.shape[1] / scale, 1.0)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    out = cv2.addWeighted(img, 4, blur, -4, 128)
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess(
    src: ArrayLike,
    img_size: int = config.IMG_SIZE,
    enhance: bool = True,
) -> np.ndarray:
    """Full pipeline -> square uint8 RGB image ready for display and tensoring."""
    img = load_rgb(src)
    img = crop_to_fundus(img)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    if enhance:
        img = ben_graham(img)
    img = circular_mask(img)
    return img


# --------------------------------------------------------------------------
# Quality assessment
# --------------------------------------------------------------------------
def quality_check(src: ArrayLike) -> dict:
    """Heuristic gate for capture quality. Returns metrics + human warnings."""
    img = load_rgb(src)
    img = crop_to_fundus(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    fov_ratio = float((gray > 15).mean())
    r, g, b = [float(img[:, :, i].mean()) for i in range(3)]
    redness = r - b  # retinas are strongly red-dominant

    warnings: list[str] = []
    critical = False

    if brightness < config.QC_MIN_BRIGHTNESS:
        warnings.append("Image looks too dark - improve illumination.")
        critical = True
    elif brightness > config.QC_MAX_BRIGHTNESS:
        warnings.append("Image looks over-exposed / washed out.")

    if lap_var < config.QC_MIN_LAPLACIAN_VAR:
        warnings.append("Image looks blurry / out of focus - hold steady and refocus.")

    if fov_ratio < config.QC_MIN_FOV_RATIO:
        warnings.append("Retina fills very little of the frame - move closer / centre the disc.")
        critical = True

    if redness < 5:
        warnings.append("Colour does not look like a typical fundus - check the lens/adapter.")

    # 0..100 quality score, blended from the three main signals.
    focus_s = min(lap_var / (config.QC_MIN_LAPLACIAN_VAR * 4), 1.0)
    bright_s = 1.0 - abs(brightness - 120) / 120
    fov_s = min(fov_ratio / 0.5, 1.0)
    score = int(max(0.0, min(1.0, 0.5 * focus_s + 0.2 * max(bright_s, 0) + 0.3 * fov_s)) * 100)

    return {
        "ok": not critical,
        "score": score,
        "sharpness": round(lap_var, 1),
        "brightness": round(brightness, 1),
        "fov_ratio": round(fov_ratio, 3),
        "redness": round(redness, 1),
        "warnings": warnings,
    }
