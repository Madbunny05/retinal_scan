"""
Multi-condition screening from the SAME fundus photo.

Rationale
---------
In a rural screening camp the patient may never see an ophthalmologist again.
The fundus image we already captured for diabetic retinopathy also carries
signal for three other common causes of avoidable blindness, and all three are
measurable with classical computer vision - no extra model, no extra capture:

  1. Glaucoma suspect        -> vertical cup-to-disc ratio (VCDR)
  2. Hypertensive retinopathy-> arteriole-to-venule ratio (AVR) + tortuosity
  3. Cataract / media haze   -> media clarity (image-domain opacity proxy)

Deliberate honesty
------------------
These are *classical morphometric estimates*, not trained models. They are
tuned to be reasonably SENSITIVE (better to over-refer than to miss) and every
finding carries an explicit confidence level. Nothing here is a diagnosis, and
the UI/report say so. This module never raises: on any failure a finding is
returned with status "unavailable" so a live demo cannot break.

Pure numpy + OpenCV, importable without torch.

`screen(img_rgb)` is the single entry point.
"""
from __future__ import annotations

import cv2
import numpy as np

import config

_SIZE = 512


# --------------------------------------------------------------------------
# Shared geometry helpers
# --------------------------------------------------------------------------
def _fov_mask(gray: np.ndarray) -> np.ndarray:
    """
    Circular retina field-of-view, minus the dark surround.

    Deliberately *inside* the mask that `preprocessing.preprocess` applies (it
    fills outside 0.96 of the radius with neutral grey). If we analysed right up
    to that boundary, the grey-vs-retina step edge would be the strongest
    gradient in the frame and would corrupt both the clarity score and the
    vessel calibre measurements.
    """
    h, w = gray.shape
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) / 2 * 0.92), 255, -1)
    mask[gray <= 12] = 0
    return mask


def _locate_disc(green: np.ndarray, fov: np.ndarray) -> tuple[tuple[int, int], int]:
    """
    Optic disc centre + approximate radius.

    The disc is the brightest large structure in the green channel. We blur
    heavily so individual exudates cannot win, then take the brightest point.
    """
    h, w = green.shape
    sigma = max(w / 25.0, 1.0)
    blurred = cv2.GaussianBlur(green, (0, 0), sigmaX=sigma)
    _, _, _, max_loc = cv2.minMaxLoc(blurred, mask=fov)

    # Refine the radius by growing a threshold region around that point.
    cx, cy = max_loc
    r_guess = int(min(h, w) * 0.11)
    x0, x1 = max(cx - 3 * r_guess, 0), min(cx + 3 * r_guess, w)
    y0, y1 = max(cy - 3 * r_guess, 0), min(cy + 3 * r_guess, h)
    patch = green[y0:y1, x0:x1]
    if patch.size < 100:
        return max_loc, r_guess

    thr = float(np.percentile(patch, 92))
    bright = (patch >= thr).astype(np.uint8) * 255
    bright = cv2.morphologyEx(
        bright, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return max_loc, r_guess

    best = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < 30:
        return max_loc, r_guess
    r_fit = int(np.sqrt(area / np.pi))
    r_fit = int(np.clip(r_fit, min(h, w) * 0.05, min(h, w) * 0.20))
    return max_loc, r_fit


def _vessel_mask(green: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """
    Vessel enhancement by black-top-hat on the inverted green channel.

    Vessels are dark, elongated and thin in the green channel; a morphological
    top-hat with a modest kernel keeps them and suppresses slow illumination
    gradients.
    """
    g = cv2.GaussianBlur(green, (0, 0), sigmaX=1.0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    g = clahe.apply(g)
    inv = 255 - g
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    top = cv2.morphologyEx(inv, cv2.MORPH_TOPHAT, k)
    thr = float(top.mean() + 2.0 * top.std())
    mask = ((top > thr).astype(np.uint8)) * 255
    mask = cv2.bitwise_and(mask, fov)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask


# --------------------------------------------------------------------------
# 1. Glaucoma suspect  -  vertical cup-to-disc ratio
# --------------------------------------------------------------------------
def _cup_to_disc(green: np.ndarray, od_center, od_radius) -> dict:
    """
    Estimate the vertical cup-to-disc ratio (VCDR).

    Inside the optic disc, the *cup* is the pale central depression - brighter
    than the surrounding neuroretinal rim. We threshold within the disc region
    at a high percentile to separate cup from rim, then compare vertical
    extents. VCDR >= ~0.6 is a widely used glaucoma-suspect referral threshold.
    """
    h, w = green.shape
    cx, cy = od_center
    pad = int(od_radius * 1.6)
    x0, x1 = max(cx - pad, 0), min(cx + pad, w)
    y0, y1 = max(cy - pad, 0), min(cy + pad, h)
    patch = green[y0:y1, x0:x1]
    if patch.size < 200:
        raise ValueError("optic disc region too small to measure")

    # Disc region: bright relative to the local patch.
    disc_thr = float(np.percentile(patch, 75))
    disc = (patch >= disc_thr).astype(np.uint8)
    disc = cv2.morphologyEx(
        disc, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

    # Cup: the brightest core within the disc.
    disc_px = patch[disc > 0]
    if disc_px.size < 100:
        raise ValueError("could not segment optic disc")
    cup_thr = float(np.percentile(disc_px, 88))
    cup = ((patch >= cup_thr) & (disc > 0)).astype(np.uint8)
    cup = cv2.morphologyEx(
        cup, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    def _vertical_extent(binary: np.ndarray) -> int:
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0
        big = max(cnts, key=cv2.contourArea)
        _, _, _, bh = cv2.boundingRect(big)
        return int(bh)

    disc_v = _vertical_extent(disc)
    cup_v = _vertical_extent(cup)
    if disc_v < 5:
        raise ValueError("optic disc vertical extent unmeasurable")

    vcdr = float(np.clip(cup_v / disc_v, 0.0, 0.95))

    # Rim thinnest sector (ISNT-rule flavour): report which side is thinnest.
    return {
        "vcdr": round(vcdr, 2),
        "disc_diameter_px": disc_v,
        "cup_diameter_px": cup_v,
        "cup_mask": cup,
        "patch_origin": (x0, y0),
    }


def _glaucoma_finding(vcdr: float) -> dict:
    if vcdr >= config.GLAUCOMA_VCDR_REFER:
        return {
            "status": "refer",
            "severity": "high",
            "headline": f"Glaucoma suspect - enlarged optic cup (VCDR {vcdr:.2f})",
            "detail": (
                "The optic nerve cup looks large relative to the disc, which can "
                "indicate glaucomatous nerve damage. Glaucoma is painless and "
                "irreversible, so this warrants an eye-pressure check and formal "
                "optic-nerve assessment."
            ),
        }
    if vcdr >= config.GLAUCOMA_VCDR_BORDERLINE:
        return {
            "status": "borderline",
            "severity": "moderate",
            "headline": f"Borderline optic cup size (VCDR {vcdr:.2f})",
            "detail": (
                "The cup-to-disc ratio is at the upper end of normal. Not "
                "alarming on its own, but worth an eye-pressure check at the "
                "next visit, especially with a family history of glaucoma."
            ),
        }
    return {
        "status": "normal",
        "severity": "low",
        "headline": f"Optic nerve appears within normal limits (VCDR {vcdr:.2f})",
        "detail": "No cup enlargement suggestive of glaucoma in this image.",
    }


# --------------------------------------------------------------------------
# 2. Hypertensive retinopathy  -  AVR + tortuosity
# --------------------------------------------------------------------------
def _vessel_metrics(img: np.ndarray, green: np.ndarray, fov: np.ndarray,
                    od_center, od_radius) -> dict:
    """
    Measure vessel calibre and tortuosity in the peripapillary zone.

    Arteries carry oxygenated blood so they are *brighter / less saturated red*
    than veins in a colour fundus photo. We split vessel segments by their mean
    red-channel intensity and compare the calibre of the widest arteriolar and
    venular segments to get an arteriole-to-venule ratio (AVR). Healthy AVR is
    ~0.66; generalised arteriolar narrowing from chronic hypertension drives it
    down.
    """
    h, w = green.shape
    vessels = _vessel_mask(green, fov)

    # Measure in an annulus around the disc (standard AVR zone: 2-3 disc
    # diameters from the disc margin), where vessels are large enough to gauge.
    zone = np.zeros_like(vessels)
    cv2.circle(zone, od_center, int(od_radius * 3.2), 255, -1)
    cv2.circle(zone, od_center, int(od_radius * 1.2), 0, -1)
    zone = cv2.bitwise_and(zone, fov)
    vz = cv2.bitwise_and(vessels, zone)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(vz, connectivity=8)
    red_ch = img[:, :, 0].astype(np.float32)

    # Calibre from the distance transform, not from area/bbox. The distance
    # transform gives each pixel its distance to the nearest non-vessel pixel, so
    # the maximum inside a segment is the half-width of the thickest part - an
    # estimate that is independent of how long or how curved the segment is.
    # (Deriving calibre from area/length and then length from area/calibre is
    # circular: it makes every tortuosity come out as exactly 1.0.)
    dist = cv2.distanceTransform(vz, cv2.DIST_L2, 3)

    # Collect plausible vessel segments first, then split them into arterioles
    # and venules *relative to each other*. An absolute intensity threshold
    # would misclassify every vessel in a dark or over-exposed capture.
    segments: list[tuple[float, float, float]] = []   # (calibre, mean_red, tortuosity)
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 40:                       # noise / lesion speck
            continue
        seg = (labels == i)
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        chord = float(np.hypot(bw, bh))     # straight-line extent of the segment
        if chord < 12:
            continue
        calibre = 2.0 * float(dist[seg].max())
        if not (0.8 <= calibre <= 14.0):    # implausible for a vessel
            continue

        # Tortuosity: distance travelled along the vessel vs the straight line
        # between its ends. A perfectly straight vessel scores 1.0; a corkscrew
        # arteriole scores well above it.
        centreline = area / max(calibre, 1e-6)
        tort = float(np.clip(centreline / max(chord, 1e-6), 1.0, 3.0))
        segments.append((calibre, float(red_ch[seg].mean()), tort))

    if len(segments) < 4:
        raise ValueError("too few measurable vessel segments")

    # Arteries carry oxygenated blood, so they read brighter in the red channel
    # than the veins in the *same* image. Split at the median.
    split = float(np.median([s[1] for s in segments]))
    arteriole_w = [s[0] for s in segments if s[1] >= split]
    venule_w = [s[0] for s in segments if s[1] < split]
    tortuosities = [s[2] for s in segments]

    if len(arteriole_w) < 2 or len(venule_w) < 2:
        raise ValueError("could not separate arterioles from venules")

    # Use the widest few of each class (as clinical AVR formulas do).
    a = float(np.mean(sorted(arteriole_w, reverse=True)[:3]))
    v = float(np.mean(sorted(venule_w, reverse=True)[:3]))
    avr = float(np.clip(a / max(v, 1e-6), 0.15, 1.4))
    tort = float(np.median(tortuosities)) if tortuosities else 1.0

    return {
        "avr": round(avr, 2),
        "tortuosity_index": round(tort, 2),
        "n_arteriole_segments": len(arteriole_w),
        "n_venule_segments": len(venule_w),
        "vessel_mask": vessels,
        "zone_mask": zone,
    }


def _hypertensive_finding(avr: float, tort: float) -> dict:
    narrowed = avr <= config.AVR_REFER
    borderline = avr <= config.AVR_BORDERLINE
    tortuous = tort >= config.TORTUOSITY_FLAG

    if narrowed:
        return {
            "status": "refer",
            "severity": "high",
            "headline": f"Arteriolar narrowing - possible hypertensive retinopathy (AVR {avr:.2f})",
            "detail": (
                "The retinal arterioles look narrow relative to the veins, a "
                "recognised sign of sustained high blood pressure. Blood-pressure "
                "measurement and management are advised - this also accelerates "
                "diabetic eye damage."
            ),
        }
    if borderline or tortuous:
        bits = []
        if borderline:
            bits.append(f"AVR {avr:.2f} is borderline low")
        if tortuous:
            bits.append(f"vessels appear tortuous (index {tort:.2f})")
        return {
            "status": "borderline",
            "severity": "moderate",
            "headline": "Possible vascular changes - check blood pressure",
            "detail": (
                " and ".join(bits).capitalize() +
                ". Worth a blood-pressure reading; controlling BP protects both "
                "the retina and the kidneys."
            ),
        }
    return {
        "status": "normal",
        "severity": "low",
        "headline": f"Retinal vessel calibre appears normal (AVR {avr:.2f})",
        "detail": "No generalised arteriolar narrowing detected in this image.",
    }


# --------------------------------------------------------------------------
# 3. Cataract / media opacity  -  media clarity
# --------------------------------------------------------------------------
def _media_clarity(img: np.ndarray, green: np.ndarray, fov: np.ndarray) -> dict:
    """
    Media clarity proxy.

    A cataract or vitreous haze scatters light on the way in and out, so the
    fundus image loses fine vessel detail, loses colour saturation and gains a
    uniform veiling glare. We combine three image-domain cues:

      * high-frequency energy inside the FOV (vessel detail retained)
      * colour saturation (haze washes colour out)
      * local contrast spread

    Note: blur from a shaky hand looks similar. We therefore only *flag* poor
    clarity and explicitly ask the operator to rule out a bad capture first -
    which is exactly what the quality check is for.
    """
    ret = fov > 0
    if ret.sum() < 500:
        raise ValueError("field of view too small to judge clarity")

    # High-frequency detail retained.
    lap = cv2.Laplacian(green, cv2.CV_64F)
    detail = float(lap[ret].var())
    detail_s = float(np.clip(detail / 350.0, 0.0, 1.0))

    # Saturation.
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = float(hsv[:, :, 1][ret].mean())
    sat_s = float(np.clip(sat / 110.0, 0.0, 1.0))

    # Local contrast spread.
    spread = float(green[ret].std())
    spread_s = float(np.clip(spread / 45.0, 0.0, 1.0))

    clarity = int(round(100 * (0.5 * detail_s + 0.25 * sat_s + 0.25 * spread_s)))
    return {
        "clarity": clarity,
        "detail_var": round(detail, 1),
        "saturation": round(sat, 1),
        "contrast_spread": round(spread, 1),
    }


def _cataract_finding(clarity: int, capture_sharp_ok: bool) -> dict:
    if clarity <= config.CLARITY_REFER:
        if not capture_sharp_ok:
            return {
                "status": "borderline",
                "severity": "moderate",
                "headline": f"Poor media clarity (score {clarity}/100) - recapture first",
                "detail": (
                    "Fine retinal detail is largely absent. This can mean a "
                    "cataract or vitreous haze, but an out-of-focus capture "
                    "looks the same. Retake the photo; if it stays hazy, refer "
                    "for a lens examination."
                ),
            }
        return {
            "status": "refer",
            "severity": "high",
            "headline": f"Media opacity suspected - possible cataract (clarity {clarity}/100)",
            "detail": (
                "The capture is in focus yet retinal detail is washed out, which "
                "suggests light scatter in the lens or vitreous. Cataract is "
                "surgically treatable - refer for a lens examination. It also "
                "limits how reliably diabetic retinopathy can be graded from "
                "this photo."
            ),
        }
    if clarity <= config.CLARITY_BORDERLINE:
        return {
            "status": "borderline",
            "severity": "moderate",
            "headline": f"Reduced media clarity (score {clarity}/100)",
            "detail": (
                "Some loss of fine detail. Could be early lens opacity or an "
                "imperfect capture. Note it and re-check at the next screening."
            ),
        }
    return {
        "status": "normal",
        "severity": "low",
        "headline": f"Ocular media appear clear (score {clarity}/100)",
        "detail": "Good retinal detail retained - no significant opacity suspected.",
    }


# --------------------------------------------------------------------------
# Overlay
# --------------------------------------------------------------------------
def _build_overlay(img: np.ndarray, od_center, od_radius,
                   cdr: dict | None, vm: dict | None) -> np.ndarray:
    out = img.copy()
    if vm is not None:
        vessels = vm.get("vessel_mask")
        zone = vm.get("zone_mask")
        if vessels is not None and zone is not None:
            vz = cv2.bitwise_and(vessels, zone)
            out[vz > 0] = (70, 200, 255)          # measured vessels: cyan
    # Disc + cup annotation
    cv2.circle(out, od_center, od_radius, (60, 160, 255), 2)
    if cdr is not None and cdr.get("cup_mask") is not None:
        x0, y0 = cdr["patch_origin"]
        cup = cdr["cup_mask"]
        ch, cw = cup.shape
        h, w = out.shape[:2]
        ch = min(ch, h - y0)
        cw = min(cw, w - x0)
        sub = out[y0:y0 + ch, x0:x0 + cw]
        m = cup[:ch, :cw] > 0
        sub[m] = (255, 230, 90)                    # cup: yellow
        out[y0:y0 + ch, x0:x0 + cw] = sub
    return cv2.addWeighted(img, 0.5, out, 0.5, 0)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def _unavailable(name: str, reason: str) -> dict:
    return {
        "condition": name,
        "status": "unavailable",
        "severity": "unknown",
        "headline": f"{name}: not assessable from this image",
        "detail": f"Measurement could not be completed ({reason}).",
        "metrics": {},
    }


def screen(img_rgb: np.ndarray, capture_sharp_ok: bool = True) -> dict:
    """
    Screen one fundus image for glaucoma, hypertensive retinopathy and
    media opacity.

    Parameters
    ----------
    img_rgb : HxWx3 uint8 RGB fundus image (raw/unenhanced works best - the
        Ben-Graham enhancement destroys the colour ratios AVR depends on).
    capture_sharp_ok : pass True when the capture is sharp enough to trust
        (i.e. quality["sharpness"] >= config.CLARITY_TRUST_SHARPNESS). Used to
        avoid blaming a cataract for what is really camera shake.

    Returns
    -------
    dict with "findings" (list, one per condition), "overlay", "metrics",
    and "any_referral" / "referrals" summary fields. Never raises.
    """
    try:
        img = cv2.resize(np.asarray(img_rgb), (_SIZE, _SIZE),
                         interpolation=cv2.INTER_AREA)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("expected an HxWx3 RGB image")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        green = img[:, :, 1]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        fov = _fov_mask(gray)
    except Exception as exc:
        # Nothing can be measured, but the caller must still get a well-formed
        # result: this module promises never to raise into a live demo.
        findings = [_unavailable(name, str(exc)) for name in
                    ("Glaucoma", "Hypertensive retinopathy",
                     "Cataract / media opacity")]
        return {
            "findings": findings, "metrics": {},
            "overlay": np.asarray(img_rgb),
            "overlay_legend": "No measurements could be taken from this image.",
            "referrals": [], "n_referrals": 0, "n_borderline": 0,
            "any_referral": False,
            "disclaimer": "Additional-condition screening was not possible "
                          "for this capture.",
        }

    try:
        od_center, od_radius = _locate_disc(green, fov)
    except Exception:
        od_center, od_radius = (_SIZE // 2, _SIZE // 2), int(_SIZE * 0.11)

    findings: list[dict] = []
    metrics: dict = {}

    # --- Glaucoma -------------------------------------------------------
    cdr = None
    try:
        cdr = _cup_to_disc(green, od_center, od_radius)
        f = _glaucoma_finding(cdr["vcdr"])
        f["condition"] = "Glaucoma"
        f["metrics"] = {"VCDR": cdr["vcdr"],
                        "disc height (px)": cdr["disc_diameter_px"],
                        "cup height (px)": cdr["cup_diameter_px"]}
        findings.append(f)
        metrics["vcdr"] = cdr["vcdr"]
    except Exception as exc:
        findings.append(_unavailable("Glaucoma", str(exc)))

    # --- Hypertensive retinopathy ---------------------------------------
    vm = None
    try:
        vm = _vessel_metrics(img, green, fov, od_center, od_radius)
        f = _hypertensive_finding(vm["avr"], vm["tortuosity_index"])
        f["condition"] = "Hypertensive retinopathy"
        f["metrics"] = {"AVR": vm["avr"],
                        "tortuosity index": vm["tortuosity_index"],
                        "arteriole segments": vm["n_arteriole_segments"],
                        "venule segments": vm["n_venule_segments"]}
        findings.append(f)
        metrics["avr"] = vm["avr"]
        metrics["tortuosity_index"] = vm["tortuosity_index"]
    except Exception as exc:
        findings.append(_unavailable("Hypertensive retinopathy", str(exc)))

    # --- Cataract / media opacity ---------------------------------------
    try:
        mc = _media_clarity(img, green, fov)
        f = _cataract_finding(mc["clarity"], capture_sharp_ok)
        f["condition"] = "Cataract / media opacity"
        f["metrics"] = {"clarity score": mc["clarity"],
                        "detail variance": mc["detail_var"],
                        "saturation": mc["saturation"]}
        findings.append(f)
        metrics["media_clarity"] = mc["clarity"]
    except Exception as exc:
        findings.append(_unavailable("Cataract / media opacity", str(exc)))

    try:
        overlay = _build_overlay(img, od_center, od_radius, cdr, vm)
    except Exception:
        overlay = img

    referrals = [f for f in findings if f["status"] == "refer"]
    borderline = [f for f in findings if f["status"] == "borderline"]

    return {
        "findings": findings,
        "metrics": metrics,
        "overlay": overlay,
        "overlay_legend": "Cyan = vessels measured for AVR · Blue ring = optic disc · Yellow = optic cup",
        "referrals": referrals,
        "n_referrals": len(referrals),
        "n_borderline": len(borderline),
        "any_referral": bool(referrals),
        "disclaimer": (
            "These additional findings come from classical image morphometry, "
            "not a trained diagnostic model. They are tuned to over-refer rather "
            "than miss disease, and must be confirmed by an eye examination."
        ),
    }
