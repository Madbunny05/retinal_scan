"""
Classical computer-vision lesion analysis.

Two jobs:
  1. Explainability - overlay likely red lesions (haemorrhages / micro-aneurysms)
     and bright lesions (exudates) on any fundus image, alongside the CNN.
  2. Fallback grader - when no trained CNN checkpoint is available, this produces
     a *deterministic, feature-based* severity estimate so the demo still works
     and is explainable, instead of returning random numbers.

This is intentionally simple (pre-deep-learning style) and is NOT a clinical
grader, but it reacts to real image content (lesion counts / areas), which is
far more honest for a demo than a random mock.
"""
from __future__ import annotations

import cv2
import numpy as np

import config

_ANALYSIS_SIZE = 512


def _fov_mask(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) / 2 * 0.97), 255, -1)
    mask[gray <= 12] = 0
    return mask


def _optic_disc(green: np.ndarray, fov: np.ndarray) -> tuple[tuple[int, int], int]:
    h, w = green.shape
    sigma = max(w / 25.0, 1.0)
    blurred = cv2.GaussianBlur(green, (0, 0), sigmaX=sigma)
    _, _, _, max_loc = cv2.minMaxLoc(blurred, mask=fov)
    radius = int(min(h, w) * 0.13)
    return max_loc, radius


def _circularity(cnt) -> float:
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return 0.0
    return float(4 * np.pi * area / (peri * peri))


def analyze(img_rgb: np.ndarray) -> dict:
    """Detect lesions and produce a deterministic severity estimate."""
    img = cv2.resize(img_rgb, (_ANALYSIS_SIZE, _ANALYSIS_SIZE), interpolation=cv2.INTER_AREA)
    green = img[:, :, 1]
    red = img[:, :, 0]
    fov = _fov_mask(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    fov_area = max(int((fov > 0).sum()), 1)

    od_center, od_radius = _optic_disc(green, fov)
    od_mask = np.zeros_like(green)
    cv2.circle(od_mask, od_center, od_radius, 255, -1)
    valid = cv2.bitwise_and(fov, cv2.bitwise_not(od_mask))

    # ---- Bright lesions (exudates) -------------------------------------
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ANALYSIS_SIZE // 20 | 1,) * 2)
    tophat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, k_big)
    th = tophat.mean() + 3.0 * tophat.std()
    exudate = ((tophat > th) & (red > 120)).astype(np.uint8) * 255
    exudate = cv2.bitwise_and(exudate, valid)
    exudate = cv2.morphologyEx(exudate, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    exudate_area = int((exudate > 0).sum())

    # ---- Red lesions (haemorrhages / micro-aneurysms) ------------------
    inv = 255 - green
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat_r = cv2.morphologyEx(inv, cv2.MORPH_TOPHAT, k_small)
    thr = tophat_r.mean() + 3.5 * tophat_r.std()
    cand = ((tophat_r > thr)).astype(np.uint8) * 255
    cand = cv2.bitwise_and(cand, valid)

    red_mask = np.zeros_like(cand)
    contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 4 <= area <= 400 and _circularity(cnt) >= 0.35:   # round, small => lesion, not vessel
            red_count += 1
            cv2.drawContours(red_mask, [cnt], -1, 255, -1)
    red_area = int((red_mask > 0).sum())

    # ---- Deterministic severity estimate -------------------------------
    ex_frac = exudate_area / fov_area
    s = 0.0
    s += min(red_count / 8.0, 2.2)          # early grades driven by red-lesion burden
    s += min(ex_frac / 0.004, 1.0) * 1.4    # exudates push toward moderate/severe
    s += min(red_area / fov_area / 0.01, 1.0) * 0.6
    s = float(np.clip(s, 0.0, 4.0))

    idx = np.arange(config.NUM_CLASSES, dtype=np.float32)
    logits = -((idx - s) ** 2) / 0.5
    probs = np.exp(logits - logits.max())
    probs = (probs / probs.sum()).tolist()
    grade = int(round(s))

    # ---- Overlay -------------------------------------------------------
    overlay = img.copy()
    overlay[red_mask > 0] = (255, 40, 40)
    overlay[exudate > 0] = (255, 235, 60)
    cv2.circle(overlay, od_center, od_radius, (60, 160, 255), 2)
    blended = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)

    return {
        "grade": grade,
        "severity_score": round(s, 2),
        "probs": probs,
        "red_lesion_count": red_count,
        "exudate_area_frac": round(ex_frac, 5),
        "overlay": blended,
        "red_mask": red_mask,
        "exudate_mask": exudate,
    }
