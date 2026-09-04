"""
Out-of-distribution (OOD) detection - "is this even a fundus photo I can grade?"

Two failure modes worth catching
--------------------------------
1. **Not a fundus image at all.** A palm, a ceiling, a document, a slide from
   the projector. A softmax classifier will happily assign one of its five DR
   grades to any of these with high confidence, because softmax is a *relative*
   score over the classes it knows - it has no way to say "none of the above".

2. **A fundus image from an unfamiliar distribution.** A different camera, a
   different pupil-dilation protocol, a different population. This is the
   documented reason field-deployed DR models underperform their published
   numbers, and it is much harder to see because the image looks fine to a human.

Two detectors, matching those two modes
---------------------------------------
* `fundus_likelihood()` - classical, always available, no model needed. Scores
  the structural signature of a retinal photograph: red-channel dominance, a
  circular bright field on a dark surround, presence of elongated dark vessel
  structures, and a detectable optic disc. Catches mode (1) robustly.

* `mahalanobis_score()` - optional. If `models/ood_stats.npz` exists (written by
  train.py), we compare the CNN's penultimate-layer feature vector against the
  training feature distribution using a Mahalanobis distance with a shared
  covariance. Large distance = the network is extrapolating. Catches mode (2).

Honest framing: the classical detector is the one that will actually fire in a
demo. The Mahalanobis path is the principled answer for a real deployment and
degrades gracefully to "unavailable" when the stats file is absent.

`check()` is the single entry point. It never raises.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import config

_SIZE = 384
_STATS_CACHE: dict | None = None
_STATS_TRIED = False


# --------------------------------------------------------------------------
# 1. Classical fundus-likelihood
# --------------------------------------------------------------------------
def fundus_likelihood(img_rgb: np.ndarray) -> dict:
    """
    Score how much this image looks like a retinal fundus photograph.

    Returns a 0-1 likelihood plus the individual cue scores, so the UI can say
    *which* cue failed rather than just refusing.
    """
    img = cv2.resize(img_rgb, (_SIZE, _SIZE), interpolation=cv2.INTER_AREA)
    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    cues: dict[str, float] = {}

    # -- Is this frame already cropped and padded? -----------------------
    # `preprocessing.preprocess` fills outside the fundus circle with flat
    # neutral grey. That is our own artefact, not evidence about the image: a
    # padded frame has no dark corners and no visible aperture, so scoring those
    # two geometry cues as 0 would penalise every properly preprocessed retina.
    # Detect the pad (near-constant, non-black surround) and mark the geometry
    # cues uninformative instead of failed.
    c = _SIZE // 8
    corners = np.concatenate([
        gray[:c, :c].ravel(), gray[:c, -c:].ravel(),
        gray[-c:, :c].ravel(), gray[-c:, -c:].ravel(),
    ])
    padded = bool(corners.std() < 3.0 and corners.mean() > 20)

    # -- Cue A: red dominance ------------------------------------------
    # Retinas are strongly red-dominant because of choroidal blood; R > G > B
    # with a wide R-B gap is very characteristic.
    lit = gray > 20
    if padded:
        # Measure colour inside the true field only; flat grey padding is
        # colour-neutral and would drag the red/blue gap towards zero.
        inner = np.zeros_like(gray)
        cv2.circle(inner, (_SIZE // 2, _SIZE // 2), int(_SIZE / 2 * 0.90), 255, -1)
        lit = lit & (inner > 0)
    if lit.sum() < 200:
        cues["red_dominance"] = 0.0
        cues["channel_order"] = 0.0
    else:
        rb_gap = float(r[lit].mean() - b[lit].mean())
        cues["red_dominance"] = float(np.clip(rb_gap / 60.0, 0.0, 1.0))
        rg_gap = float(r[lit].mean() - g[lit].mean())
        cues["channel_order"] = 1.0 if (rb_gap > 8 and rg_gap > 4) else 0.0

    # -- Cue B: circular bright field on a dark surround ----------------
    # Fundus cameras image through a round aperture, so the lit region is a
    # disc centred in the frame with dark corners.
    if padded:
        cues["circular_field"] = 0.5
        cues["dark_surround"] = 0.5
    else:
        fg = (gray > 20).astype(np.uint8)
        fg = cv2.morphologyEx(
            fg, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circularity = 0.0
        if cnts:
            big = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(big)
            peri = cv2.arcLength(big, True)
            if peri > 0 and area > (_SIZE * _SIZE) * 0.05:
                circularity = float(4 * np.pi * area / (peri * peri))
        cues["circular_field"] = float(np.clip(circularity / 0.85, 0.0, 1.0))

        # Corner darkness: the four corners of a fundus frame are near-black.
        centre = gray[_SIZE // 2 - c:_SIZE // 2 + c, _SIZE // 2 - c:_SIZE // 2 + c]
        corner_ratio = float(corners.mean() / max(centre.mean(), 1e-6))
        cues["dark_surround"] = float(np.clip((0.75 - corner_ratio) / 0.55, 0.0, 1.0))

    # -- Cue C: vessel-like elongated dark structures -------------------
    # The retinal vascular tree is the single most distinctive fundus feature.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    ge = clahe.apply(img[:, :, 1])
    inv = 255 - ge
    top = cv2.morphologyEx(
        inv, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    vmask = (top > (top.mean() + 2.0 * top.std())).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(vmask, connectivity=8)
    elongated = 0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 30:
            continue
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        diag = float(np.hypot(bw, bh))
        if diag <= 0:
            continue
        # thin + long => vessel-like, as opposed to a blob
        if a / diag < 6.0 and diag > 25:
            elongated += 1
    cues["vessel_structure"] = float(np.clip(elongated / 12.0, 0.0, 1.0))

    # -- Cue D: detectable optic disc ------------------------------------
    # A compact bright region noticeably brighter than the retinal background.
    blur = cv2.GaussianBlur(img[:, :, 1], (0, 0), sigmaX=max(_SIZE / 25.0, 1.0))
    fovm = np.zeros_like(gray)
    cv2.circle(fovm, (_SIZE // 2, _SIZE // 2), int(_SIZE / 2 * 0.90), 255, -1)
    fovm[gray <= 12] = 0
    if (fovm > 0).sum() > 500:
        _, mx, _, _ = cv2.minMaxLoc(blur, mask=fovm)
        bg = float(blur[fovm > 0].mean())
        disc_contrast = float(mx - bg)
        cues["optic_disc"] = float(np.clip(disc_contrast / 45.0, 0.0, 1.0))
    else:
        cues["optic_disc"] = 0.0

    # -- Weighted combination -------------------------------------------
    weights = {
        "red_dominance": 0.24,
        "channel_order": 0.10,
        "circular_field": 0.12,
        "dark_surround": 0.12,
        "vessel_structure": 0.26,
        "optic_disc": 0.16,
    }
    score = sum(cues[k] * w for k, w in weights.items())
    score = float(np.clip(score, 0.0, 1.0))

    # Which cue is most responsible for a low score? Not simply the lowest cue -
    # the one that costs the total score the most, i.e. the largest weighted
    # shortfall (1 - cue) * weight. A weak cue with a small weight matters less
    # than a middling cue carrying a quarter of the score.
    weakest = max(cues.items(), key=lambda kv: (1.0 - kv[1]) * weights[kv[0]])
    return {
        "likelihood": round(score, 3),
        "cues": {k: round(v, 3) for k, v in cues.items()},
        "weakest_cue": weakest[0],
    }


_CUE_EXPLANATION = {
    "red_dominance": "the colour balance is not red-dominant the way a retina is",
    "channel_order": "the red/green/blue balance does not match retinal tissue",
    "circular_field": "there is no circular camera aperture visible",
    "dark_surround": "the frame corners are not dark as they are in fundus photos",
    "vessel_structure": "no branching retinal vessel tree could be found",
    "optic_disc": "no optic disc could be located",
}


# --------------------------------------------------------------------------
# 2. Mahalanobis distance on CNN features (optional)
# --------------------------------------------------------------------------
def _load_stats() -> dict | None:
    """Load models/ood_stats.npz once. Returns None if absent or malformed."""
    global _STATS_CACHE, _STATS_TRIED
    if _STATS_TRIED:
        return _STATS_CACHE
    path = Path(config.OOD_STATS_PATH)
    if not path.exists():
        # Do NOT latch here: a training run may write the file while the app is
        # already open, and the operator should not have to restart to get the
        # feature-space check.
        return None
    _STATS_TRIED = True
    try:
        z = np.load(str(path))
        mean = np.asarray(z["mean"], dtype="float64")
        prec = np.asarray(z["precision"], dtype="float64")
        ref = float(z["ref_distance"]) if "ref_distance" in z else None
        if mean.ndim != 1 or prec.shape != (mean.size, mean.size):
            return None
        _STATS_CACHE = {"mean": mean, "precision": prec, "ref_distance": ref}
    except Exception:
        _STATS_CACHE = None
    return _STATS_CACHE


def mahalanobis_score(features: np.ndarray) -> dict | None:
    """
    Squared Mahalanobis distance of a feature vector from the training mean.

    Uses a class-agnostic (tied) covariance, which is the standard simple form
    of the Lee et al. 2018 detector. Returns None when no stats are available.
    """
    stats = _load_stats()
    if stats is None:
        return None
    f = np.asarray(features, dtype="float64").ravel()
    if f.size != stats["mean"].size:
        return None
    d = f - stats["mean"]
    dist = float(d @ stats["precision"] @ d)
    ref = stats.get("ref_distance")
    ratio = float(dist / ref) if ref and ref > 0 else None
    return {
        "distance": round(dist, 2),
        "ref_distance": round(ref, 2) if ref else None,
        "ratio": round(ratio, 2) if ratio is not None else None,
        "is_ood": bool(ratio is not None and ratio > config.OOD_MAHALANOBIS_RATIO),
    }


def fit_stats(features: np.ndarray, out_path: str | None = None,
              percentile: float = 97.5) -> dict:
    """
    Fit the OOD reference distribution from training features.

    Call this after training with the penultimate features of the training set.
    Saves mean, precision (inverse covariance) and a reference distance taken at
    the given percentile of in-distribution distances - so the threshold is
    expressed relative to what the model has actually seen.
    """
    X = np.asarray(features, dtype="float64")
    if X.ndim != 2 or X.shape[0] < X.shape[1]:
        # Not enough samples for a stable full covariance; shrink hard.
        pass
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    # Ledoit-Wolf-style shrinkage toward a scaled identity keeps the inverse
    # well-conditioned when features outnumber samples.
    trace_mean = float(np.trace(cov) / max(cov.shape[0], 1))
    cov_s = 0.9 * cov + 0.1 * trace_mean * np.eye(cov.shape[0])
    prec = np.linalg.pinv(cov_s)
    d = np.einsum("ij,jk,ik->i", Xc, prec, Xc)
    ref = float(np.percentile(d, percentile))
    out = {"mean": mean, "precision": prec, "ref_distance": ref}
    if out_path:
        np.savez_compressed(out_path, **out)
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def check(img_rgb: np.ndarray, features: np.ndarray | None = None) -> dict:
    """
    Decide whether this image is safe to grade.

    Parameters
    ----------
    img_rgb : raw (unenhanced) RGB fundus candidate.
    features : optional penultimate CNN feature vector for the Mahalanobis test.

    Returns
    -------
    dict with is_ood, reason, likelihood, per-cue scores and the optional
    feature-space result. Never raises - on internal failure it returns
    is_ood=False with a note, because refusing to grade every image would be
    worse than not running the check.
    """
    try:
        fl = fundus_likelihood(img_rgb)
    except Exception as exc:
        # Same key set as the success path, so consumers never need to guess
        # which branch produced the dict.
        return {
            "is_ood": False,
            "checked": False,
            "level": "unknown",
            "reason": f"OOD check unavailable ({exc}).",
            "reasons": [],
            "likelihood": None,
            "cues": {},
            "weakest_cue": None,
            "feature_space": None,
            "feature_check_available": False,
        }

    like = fl["likelihood"]
    reasons: list[str] = []
    is_ood = False
    level = "in-distribution"

    if like < config.OOD_MIN_FUNDUS_LIKELIHOOD:
        is_ood = True
        level = "not-a-fundus"
        why = _CUE_EXPLANATION.get(fl["weakest_cue"], "it lacks retinal structure")
        reasons.append(
            f"This does not look like a retinal fundus photograph "
            f"(score {like:.2f}/1.00 - {why}). Grading it would produce a "
            f"meaningless number."
        )
    elif like < config.OOD_WARN_FUNDUS_LIKELIHOOD:
        level = "atypical"
        why = _CUE_EXPLANATION.get(fl["weakest_cue"], "some retinal cues are weak")
        reasons.append(
            f"Image is an atypical fundus capture (score {like:.2f}) - {why}. "
            f"Treat the grade with caution."
        )

    fs = None
    if features is not None:
        try:
            fs = mahalanobis_score(features)
        except Exception:
            fs = None
        if fs is not None and fs.get("is_ood"):
            is_ood = True
            level = "unfamiliar-distribution"
            reasons.append(
                f"The image sits far outside the training distribution in "
                f"feature space (Mahalanobis ratio {fs['ratio']:.2f}x the "
                f"in-distribution reference) - likely a different camera or "
                f"population than the model was trained on."
            )

    return {
        "is_ood": is_ood,
        "checked": True,
        "level": level,
        "likelihood": like,
        "cues": fl["cues"],
        "weakest_cue": fl["weakest_cue"],
        "reason": " ".join(reasons) if reasons else
                  f"Image looks like a valid fundus capture (score {like:.2f}).",
        "reasons": reasons,
        "feature_space": fs,
        "feature_check_available": fs is not None,
    }
