"""
Uncertainty quantification and abstention.

Why this matters more than another accuracy point
-------------------------------------------------
A screening tool that is confidently wrong is worse than one that says "I don't
know - recapture or refer". Published DR models degrade sharply on images from
cameras and populations they were not trained on, and the failure is usually
*silent*. This module makes the model's ignorance visible and turns it into a
concrete action.

Three independent signals, then one decision
--------------------------------------------
  1. Predictive entropy      - how spread out the class distribution is.
  2. Top-2 margin            - how close the runner-up grade is.
  3. Augmentation agreement  - do flips/rotations of the same eye agree?
                               (Test-time augmentation, TTA.) Disagreement
                               across views that a clinician would grade
                               identically is strong evidence of instability.
  4. MC-dropout variance     - optional, when the backbone has dropout: keep
                               dropout active at inference and sample. High
                               variance = the model itself is unsure, as opposed
                               to the image being genuinely borderline.

`decide()` folds these into one of three outcomes:
  confident     - report the grade normally
  borderline    - report it, but flag for human review
  inconclusive  - do NOT report a grade; ask for recapture or refer

Calibration
-----------
Softmax outputs from a cross-entropy-trained net are systematically
overconfident. `apply_temperature()` implements temperature scaling (Guo et al.
2017): a single scalar T learned on a validation set that flattens the logits
without changing the argmax. `fit_temperature()` is provided so train.py can
learn T and store it beside the checkpoint. If no T is available we use 1.0 and
say so, rather than pretending the confidence is calibrated.

Pure numpy for everything except the optional torch paths.
"""
from __future__ import annotations

import numpy as np

import config


# --------------------------------------------------------------------------
# Test-time augmentation views
# --------------------------------------------------------------------------
def tta_views(img: np.ndarray, n: int | None = None) -> list[np.ndarray]:
    """
    Generate label-preserving views of a fundus image.

    Horizontal flip, vertical flip and 180-degree rotation are all valid for a
    fundus photo: DR severity does not depend on orientation. (A 90-degree
    rotation is also geometrically harmless for lesion counting, so we include
    it, but flips come first since they are the most natural.)
    """
    if n is None:
        n = config.TTA_VIEWS
    views = [img]
    candidates = [
        np.ascontiguousarray(img[:, ::-1, :]),          # horizontal flip
        np.ascontiguousarray(img[::-1, :, :]),          # vertical flip
        np.ascontiguousarray(img[::-1, ::-1, :]),       # 180 rotation
        np.ascontiguousarray(np.rot90(img, 1)),
        np.ascontiguousarray(np.rot90(img, 3)),
    ]
    for c in candidates:
        if len(views) >= max(1, n):
            break
        views.append(c)
    return views[:max(1, n)]


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """
    Re-scale a probability vector by a temperature.

    We recover pseudo-logits with log(p), divide by T and re-normalise. T > 1
    softens (reduces overconfidence); T < 1 sharpens. argmax is unchanged.
    """
    t = float(temperature)
    if not np.isfinite(t) or t <= 0 or abs(t - 1.0) < 1e-6:
        return np.asarray(probs, dtype="float64")
    p = np.clip(np.asarray(probs, dtype="float64"), 1e-12, 1.0)
    logits = np.log(p) / t
    logits -= logits.max()
    e = np.exp(logits)
    return e / e.sum()


def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    lo: float = 0.05, hi: float = 10.0, iters: int = 200) -> float:
    """
    Learn a single temperature by minimising validation NLL.

    Ternary search over T - the NLL is unimodal in T, so this converges without
    needing gradients or torch. Call this from a training script on held-out
    logits and store the result in the checkpoint.
    """
    logits = np.asarray(logits, dtype="float64")
    labels = np.asarray(labels, dtype="int64")
    if logits.ndim != 2 or logits.shape[0] != labels.shape[0] or logits.shape[0] == 0:
        return 1.0

    def nll(t: float) -> float:
        z = logits / max(t, 1e-6)
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(labels)), labels].mean())

    for _ in range(iters):
        if hi - lo < 1e-4:
            break
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
    return round((lo + hi) / 2.0, 4)


# --------------------------------------------------------------------------
# Uncertainty metrics
# --------------------------------------------------------------------------
def _entropy(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype="float64"), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def quantify(view_probs: np.ndarray, temperature: float = 1.0,
             mc_probs: np.ndarray | None = None) -> dict:
    """
    Turn a stack of per-view probability vectors into uncertainty metrics.

    Parameters
    ----------
    view_probs : (V, C) array - one row per TTA view.
    temperature : calibration scalar applied to the mean distribution.
    mc_probs : optional (S, C) array of MC-dropout samples.

    Returns
    -------
    dict of metrics plus the calibrated mean distribution.
    """
    vp = np.atleast_2d(np.asarray(view_probs, dtype="float64"))
    if vp.size == 0 or vp.shape[1] == 0:
        # Degenerate input: fall back to a maximally-uninformative distribution
        # so the caller gets metrics that correctly read as "no information".
        vp = np.full((1, config.NUM_CLASSES), 1.0 / config.NUM_CLASSES)
    vp = vp / np.clip(vp.sum(axis=1, keepdims=True), 1e-12, None)

    mean_raw = vp.mean(axis=0)
    mean_p = apply_temperature(mean_raw, temperature)

    # Rank descending. `argsort(-p, stable)` breaks exact ties towards the lower
    # index, which is what np.argmax does - the per-view argmax below must agree
    # with top1 or the agreement metric silently collapses to zero on a tie.
    order = np.argsort(-mean_p, kind="stable")
    top1, top2 = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
    margin = float(mean_p[top1] - mean_p[top2])

    ent = _entropy(mean_p)
    max_ent = float(np.log(len(mean_p)))
    norm_ent = float(ent / max_ent) if max_ent > 0 else 0.0

    # Augmentation agreement: fraction of views whose argmax matches the mean's.
    view_grades = vp.argmax(axis=1)
    agreement = float((view_grades == top1).mean())
    # Spread of the predicted grade across views, in grade units.
    view_spread = float(view_grades.max() - view_grades.min()) if len(view_grades) > 1 else 0.0

    # Expected grade under the distribution, and its standard deviation. For an
    # ordinal target this is more informative than argmax alone: a distribution
    # straddling grades 1 and 3 is qualitatively different from one on 2 alone.
    idx = np.arange(len(mean_p), dtype="float64")
    exp_grade = float((mean_p * idx).sum())
    grade_std = float(np.sqrt((mean_p * (idx - exp_grade) ** 2).sum()))

    out = {
        "probs": mean_p.tolist(),
        "probs_uncalibrated": mean_raw.tolist(),
        "grade": top1,
        "runner_up": top2,
        "confidence": float(mean_p[top1]),
        "margin": round(margin, 4),
        "entropy": round(ent, 4),
        "entropy_norm": round(norm_ent, 4),
        "tta_agreement": round(agreement, 3),
        "tta_views": int(vp.shape[0]),
        "tta_grade_spread": view_spread,
        "expected_grade": round(exp_grade, 2),
        "grade_std": round(grade_std, 3),
        "temperature": float(temperature),
        "calibrated": abs(float(temperature) - 1.0) > 1e-6,
        "mc_dropout_std": None,
        "mc_samples": 0,
    }

    if mc_probs is not None:
        mc = np.atleast_2d(np.asarray(mc_probs, dtype="float64"))
        if mc.shape[0] > 1 and mc.shape[1] > top1:
            # Variance of the top class probability across stochastic passes.
            out["mc_dropout_std"] = round(float(mc[:, top1].std()), 4)
            out["mc_samples"] = int(mc.shape[0])
            mc_grades = mc.argmax(axis=1)
            out["mc_agreement"] = round(float((mc_grades == top1).mean()), 3)

    return out


# --------------------------------------------------------------------------
# The decision
# --------------------------------------------------------------------------
def decide(metrics: dict, quality_score: int | None = None,
           ood: dict | None = None) -> dict:
    """
    Fold uncertainty (and optionally capture quality + OOD) into an action.

    Design note - why a near-tie is not automatically an abstention
    --------------------------------------------------------------
    DR grades are ordinal, and what actually happens to the patient is a
    *referral decision*, not a number. A model torn between grades 3 and 4 has
    still told us everything we need: refer urgently either way. A model torn
    between 1 and 2 has told us nothing, because that tie straddles the referral
    threshold.

    So we treat the two cases differently:
      * tie *within* a referral band -> report the higher grade, note the tie
      * tie *across* the referral band -> escalate to the more urgent grade
        (fail-safe) and flag for review, rather than reporting a coin flip

    Outright abstention is reserved for cases where nothing can be salvaged:
    an unusable capture, an out-of-distribution image, a diffuse prediction, or
    a prediction that is unstable under augmentations of the same eye.

    Returns a dict with:
      verdict : "confident" | "borderline" | "inconclusive"
      report_grade : whether the UI should show a grade at all
      effective_grade : the grade to act on (may be escalated for safety)
      reasons : human-readable list of why
      action : what the operator should do
    """
    reasons: list[str] = []
    hard = 0     # reasons to abstain outright
    soft = 0     # reasons to flag for review

    conf = metrics.get("confidence", 0.0)
    margin = metrics.get("margin", 1.0)
    ent = metrics.get("entropy_norm", 0.0)
    agree = metrics.get("tta_agreement", 1.0)
    spread = metrics.get("tta_grade_spread", 0.0)
    mc_std = metrics.get("mc_dropout_std")
    grade = int(metrics.get("grade", 0))
    runner_up = int(metrics.get("runner_up", grade))

    effective_grade = grade
    escalated = False

    if conf < config.ABSTAIN_MIN_CONFIDENCE:
        hard += 1
        reasons.append(
            f"Top-class confidence is only {conf*100:.0f}% "
            f"(needs {config.ABSTAIN_MIN_CONFIDENCE*100:.0f}%).")
    elif conf < config.REVIEW_MIN_CONFIDENCE:
        soft += 1
        reasons.append(f"Confidence {conf*100:.0f}% is modest.")

    # --- Near-tie handling, referral-band aware -------------------------
    thr = config.REFERRAL_THRESHOLD_GRADE
    same_band = (grade >= thr) == (runner_up >= thr)
    if margin < config.REVIEW_MIN_MARGIN and runner_up != grade:
        soft += 1
        if same_band:
            reasons.append(
                f"Grades {grade} and {runner_up} are close (gap "
                f"{margin*100:.0f} points), but both lead to the same referral "
                f"decision, so the tie does not change what happens next.")
        elif margin < config.ABSTAIN_MIN_MARGIN and runner_up > grade:
            # A genuine coin flip that straddles the referral line. Reporting the
            # lower grade would send a possibly-referable patient home.
            effective_grade = runner_up
            escalated = True
            reasons.append(
                f"Grades {grade} and {runner_up} are effectively tied (gap "
                f"{margin*100:.0f} points) and they straddle the referral "
                f"threshold. Acting on grade {runner_up} - the fail-safe "
                f"direction for a screening tool.")
        else:
            reasons.append(
                f"Grade {grade} is close to grade {runner_up} (gap "
                f"{margin*100:.0f} points) and the two sit on opposite sides of "
                f"the referral threshold, so the referral decision itself is "
                f"borderline.")

    if ent > config.ABSTAIN_MAX_ENTROPY:
        hard += 1
        reasons.append(f"Prediction is diffuse across grades (entropy {ent:.2f}).")
    elif ent > config.REVIEW_MAX_ENTROPY:
        soft += 1
        reasons.append(f"Prediction is somewhat spread out (entropy {ent:.2f}).")

    if agree < config.ABSTAIN_MIN_TTA_AGREEMENT:
        hard += 1
        reasons.append(
            f"Only {agree*100:.0f}% of flipped/rotated views of this same image "
            f"agree on the grade - the prediction is unstable.")
    elif agree < config.REVIEW_MIN_TTA_AGREEMENT:
        soft += 1
        reasons.append(f"Augmented views agree only {agree*100:.0f}% of the time.")

    if spread >= 2:
        hard += 1
        reasons.append(
            f"Augmented views of the same eye differ by up to {int(spread)} "
            f"grades - too unstable to report a single number.")

    if mc_std is not None and mc_std > config.ABSTAIN_MAX_MC_STD:
        hard += 1
        reasons.append(
            f"Repeated stochastic passes disagree (dropout std {mc_std:.2f}) - "
            f"the model itself is unsure.")

    if quality_score is not None and quality_score < config.ABSTAIN_MIN_QUALITY:
        hard += 1
        reasons.append(
            f"Capture quality {quality_score}/100 is below the "
            f"{config.ABSTAIN_MIN_QUALITY} needed for a reliable read.")

    if ood is not None and ood.get("is_ood"):
        hard += 1
        reasons.append(ood.get("reason", "Image looks unlike the training data."))

    if hard:
        verdict, report_grade = "inconclusive", False
        action = (
            "Do not act on a grade from this image. Retake the photo with better "
            "focus, lighting and centring. If a good capture still comes back "
            "inconclusive, refer the patient for a manual examination."
        )
    elif soft:
        verdict, report_grade = "borderline", True
        action = (
            "Report the grade but treat it as provisional: have a clinician "
            "review the image, or re-screen sooner than the routine interval."
        )
        if escalated:
            action = (
                f"Act on grade {effective_grade} (escalated for safety). Have a "
                f"clinician review the image before stepping the urgency back down."
            )
    else:
        verdict, report_grade = "confident", True
        action = "Proceed with the grade-based referral recommendation below."
        reasons.append(
            "Confidence, class margin and augmentation agreement all pass their "
            "thresholds.")

    return {
        "verdict": verdict,
        "report_grade": report_grade,
        "effective_grade": int(effective_grade),
        "escalated": escalated,
        "reasons": reasons,
        "action": action,
        "n_hard": hard,
        "n_soft": soft,
    }


# --------------------------------------------------------------------------
# Presentation helper
# --------------------------------------------------------------------------
VERDICT_STYLE = {
    "confident":    {"color": "#1a9850", "icon": "✓", "label": "Confident"},
    "borderline":   {"color": "#f0ad4e", "icon": "!", "label": "Needs review"},
    "inconclusive": {"color": "#d73027", "icon": "✕", "label": "Inconclusive"},
}


def summary_line(metrics: dict) -> str:
    """One-line technical summary for the report footer / sidebar."""
    bits = [
        f"conf {metrics.get('confidence', 0)*100:.0f}%",
        f"margin {metrics.get('margin', 0)*100:.0f}pts",
        f"entropy {metrics.get('entropy_norm', 0):.2f}",
        f"TTA agree {metrics.get('tta_agreement', 0)*100:.0f}% "
        f"({metrics.get('tta_views', 1)} views)",
    ]
    if metrics.get("mc_dropout_std") is not None:
        bits.append(f"MC-dropout std {metrics['mc_dropout_std']:.2f}")
    bits.append("calibrated" if metrics.get("calibrated") else "uncalibrated (T=1)")
    return " · ".join(bits)
