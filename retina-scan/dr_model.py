"""
Model layer for RetinaScan.

Design goals for a hackathon:
  * Real architecture (EfficientNet via `timm`) with genuine 5-class ICDR head.
  * Works with a trained checkpoint if you have one (fully offline).
  * NEVER crashes a live demo: if torch/timm/weights are missing it transparently
    falls back to the deterministic classical-CV grader in `lesion_analysis`.

`run_inference()` is the single entry point the UI calls.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import config
import preprocessing
import lesion_analysis
import multi_disease
import ood
import uncertainty

# torch / timm are imported lazily so the rest of the app (and the CV fallback)
# still work even if they are not installed.
_torch = None
_timm = None
_MODEL: "DRModel | None" = None


def _lazy_torch():
    global _torch, _timm
    if _torch is None:
        import torch  # noqa
        import timm    # noqa
        _torch, _timm = torch, timm
    return _torch, _timm


def _clean_key(k: str) -> str:
    for prefix in ("module.", "model."):
        if k.startswith(prefix):
            k = k[len(prefix):]
    return k


def _find_checkpoint() -> "Path | None":
    if config.LOCAL_CHECKPOINT.exists():
        return config.LOCAL_CHECKPOINT
    if config.HF_REPO:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=config.HF_REPO, filename=config.HF_FILE)
            return Path(path)
        except Exception:
            return None
    return None


class DRModel:
    """Loads a trained CNN if possible; otherwise stays in classical demo mode."""

    def __init__(self) -> None:
        self.mode = "demo"                 # "trained" | "demo"
        self.net = None
        self.device = "cpu"
        self.source = "classical CV (no trained checkpoint found)"
        # Temperature-scaling scalar. Stays at the default (1.0 = uncalibrated)
        # unless the checkpoint carries a value learned on a validation set.
        self.temperature = float(config.DEFAULT_TEMPERATURE)
        self.has_dropout = False

    def load(self) -> "DRModel":
        try:
            torch, timm = _lazy_torch()
        except Exception as exc:  # torch/timm not installed
            self.mode = "demo"
            self.source = f"classical CV (PyTorch/timm unavailable: {exc})"
            return self

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = _find_checkpoint()
        if ckpt is None:
            # Intentionally NOT using an untrained ImageNet head here: predicting
            # DR grades from random weights would be misleading. Use CV fallback.
            self.mode = "demo"
            return self

        try:
            net = timm.create_model(
                config.BACKBONE, pretrained=False, num_classes=config.NUM_CLASSES
            )
            state = torch.load(str(ckpt), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                # A richer checkpoint may also carry the calibration scalar.
                temp = state.get("temperature")
                if temp is not None:
                    try:
                        self.temperature = float(temp)
                    except Exception:
                        pass
                state = state["state_dict"]
            state = {_clean_key(k): v for k, v in state.items()}
            net.load_state_dict(state, strict=False)
            net.eval().to(self.device)
            self.net = net
            self.mode = "trained"
            self.has_dropout = any(
                isinstance(m, torch.nn.modules.dropout._DropoutNd)
                for m in net.modules()
            )
            cal = ("T=%.3f" % self.temperature) if self.temperature != 1.0 else "uncalibrated"
            self.source = (f"trained checkpoint: {ckpt.name} "
                           f"({config.BACKBONE}, {cal})")
        except Exception as exc:
            self.mode = "demo"
            self.source = f"classical CV (checkpoint load failed: {exc})"
        return self

    # -- tensor helpers ----------------------------------------------------
    def _to_tensor(self, rgb: np.ndarray):
        torch, _ = _lazy_torch()
        x = rgb.astype("float32") / 255.0
        mean = np.array(config.IMAGENET_MEAN, dtype="float32")
        std = np.array(config.IMAGENET_STD, dtype="float32")
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))[None]
        return torch.from_numpy(np.ascontiguousarray(x)).to(self.device)

    # -- inference ---------------------------------------------------------
    def predict_cnn(self, enhanced_rgb: np.ndarray) -> np.ndarray:
        torch, _ = _lazy_torch()
        with torch.no_grad():
            logits = self.net(self._to_tensor(enhanced_rgb))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return probs

    def predict_tta(self, enhanced_rgb: np.ndarray,
                    n_views: int | None = None) -> np.ndarray:
        """
        Predict over several label-preserving views of the same image.

        Returns a (V, C) array - one probability row per view. Averaging these
        gives a steadier estimate; their *disagreement* is the stability signal
        that drives abstention.
        """
        torch, _ = _lazy_torch()
        views = uncertainty.tta_views(enhanced_rgb, n_views)
        rows = []
        with torch.no_grad():
            for v in views:
                logits = self.net(self._to_tensor(v))
                rows.append(torch.softmax(logits, dim=1)[0].cpu().numpy())
        return np.stack(rows, axis=0)

    def predict_mc_dropout(self, enhanced_rgb: np.ndarray,
                           n_samples: int | None = None) -> "np.ndarray | None":
        """
        Sample the predictive distribution with dropout left ON at inference.

        This approximates Bayesian model uncertainty: if repeated stochastic
        passes over the *same* pixels disagree, the model's own parameters are
        unsure, which is different from the image being genuinely borderline.
        Returns None when the backbone has no dropout layers.
        """
        if not self.has_dropout:
            return None
        n = config.MC_DROPOUT_SAMPLES if n_samples is None else n_samples
        if n <= 1:
            return None
        torch, _ = _lazy_torch()

        # Enable ONLY dropout; BatchNorm must stay in eval mode or its running
        # statistics get polluted and the samples become meaningless.
        was_training = self.net.training
        self.net.eval()
        dropouts = [m for m in self.net.modules()
                    if isinstance(m, torch.nn.modules.dropout._DropoutNd)]
        for m in dropouts:
            m.train()
        try:
            x = self._to_tensor(enhanced_rgb)
            rows = []
            with torch.no_grad():
                for _ in range(int(n)):
                    logits = self.net(x)
                    rows.append(torch.softmax(logits, dim=1)[0].cpu().numpy())
            return np.stack(rows, axis=0)
        except Exception:
            return None
        finally:
            for m in dropouts:
                m.eval()
            self.net.train(was_training)

    def features(self, enhanced_rgb: np.ndarray) -> "np.ndarray | None":
        """
        Penultimate-layer feature vector, for the feature-space OOD test.

        Global-average-pools the final conv feature map so the vector length is
        the backbone's channel count regardless of input resolution.
        """
        torch, _ = _lazy_torch()
        try:
            with torch.no_grad():
                feats = self.net.forward_features(self._to_tensor(enhanced_rgb))
                if feats.ndim == 4:
                    feats = feats.mean(dim=(2, 3))
                elif feats.ndim == 3:
                    feats = feats.mean(dim=1)
                return feats[0].cpu().numpy().astype("float64")
        except Exception:
            return None

    def gradcam(self, enhanced_rgb: np.ndarray, class_idx: int) -> np.ndarray:
        """Grad-CAM overlay on the final conv feature map."""
        torch, _ = _lazy_torch()
        x = self._to_tensor(enhanced_rgb)
        x.requires_grad_(True)
        with torch.enable_grad():
            feats = self.net.forward_features(x)
            feats.retain_grad()
            logits = self.net.forward_head(feats)
            score = logits[0, class_idx]
            self.net.zero_grad(set_to_none=True)
            score.backward()
        grads = feats.grad
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feats).sum(dim=1, keepdim=True))[0, 0]
        cam = cam.detach().cpu().numpy().astype("float32")
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cam = cv2.resize(cam, (enhanced_rgb.shape[1], enhanced_rgb.shape[0]))
        heat = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(enhanced_rgb, 0.55, heat, 0.45, 0)


def get_model() -> DRModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = DRModel().load()
    return _MODEL


def _demo_view_probs(raw: np.ndarray, lesion: dict | None) -> np.ndarray:
    """
    Build a (V, C) stack for demo mode by re-running the classical grader on
    augmented views.

    The classical grader is deterministic per-image but NOT invariant to flips
    (thresholds are computed from local statistics), so this yields a genuine
    stability signal in demo mode too, rather than faking one.

    Row 0 is always the un-augmented view, which is what the UI labels
    "original" - so if the base analysis failed there is nothing to align to and
    we return a single uninformative row rather than silently relabelling the
    h-flip as the original.
    """
    if lesion is None:
        return np.full((1, config.NUM_CLASSES), 1.0 / config.NUM_CLASSES)
    rows = [np.asarray(lesion["probs"], dtype="float64")]
    for v in uncertainty.tta_views(raw)[1:]:
        try:
            rows.append(np.asarray(lesion_analysis.analyze(v)["probs"], dtype="float64"))
        except Exception:
            continue
    return np.stack(rows, axis=0)


def run_inference(original_rgb: np.ndarray, enhance: bool = True,
                  use_tta: bool = True, use_mc_dropout: bool = True,
                  quality_score: int | None = None,
                  sharpness: float | None = None,
                  screen_other_conditions: bool = True) -> dict:
    """
    End-to-end: preprocess -> OOD gate -> predict (TTA) -> quantify uncertainty
    -> abstain or report -> explain -> screen other conditions.

    The UI calls only this. Every added stage is individually failure-tolerant:
    if uncertainty or multi-disease screening throws, the core grade still comes
    back.
    """
    model = get_model()
    enhanced = preprocessing.preprocess(original_rgb, enhance=enhance)
    raw = preprocessing.preprocess(original_rgb, enhance=False)

    lesion = None
    try:
        lesion = lesion_analysis.analyze(raw)
    except Exception:
        lesion = None

    # ---- Prediction (+ TTA views) --------------------------------------
    mc_probs = None
    feats = None
    if model.mode == "trained":
        try:
            view_probs = (model.predict_tta(enhanced) if use_tta
                          else model.predict_cnn(enhanced)[None, :])
        except Exception:
            # The recovery path must be guarded too - this function is the one
            # thing in the app that is not allowed to fail.
            try:
                view_probs = model.predict_cnn(enhanced)[None, :]
            except Exception:
                view_probs = _demo_view_probs(raw, lesion)
        if use_mc_dropout:
            try:
                mc_probs = model.predict_mc_dropout(enhanced)
            except Exception:
                mc_probs = None
        feats = model.features(enhanced)
    else:
        view_probs = (_demo_view_probs(raw, lesion) if use_tta
                      else np.asarray(
                          [lesion["probs"]] if lesion else
                          [[1.0 / config.NUM_CLASSES] * config.NUM_CLASSES],
                          dtype="float64"))

    # ---- Uncertainty ---------------------------------------------------
    try:
        unc = uncertainty.quantify(view_probs, temperature=model.temperature,
                                   mc_probs=mc_probs)
    except Exception:
        mean_p = np.atleast_2d(view_probs).mean(axis=0)
        unc = {"probs": [float(p) for p in mean_p],
               "grade": int(np.argmax(mean_p)),
               "confidence": float(np.max(mean_p)),
               "margin": 1.0, "entropy_norm": 0.0, "tta_agreement": 1.0,
               "tta_views": 1, "tta_grade_spread": 0.0, "temperature": 1.0,
               "calibrated": False, "mc_dropout_std": None}

    probs = np.asarray(unc["probs"], dtype="float64")
    grade = int(np.clip(unc["grade"], 0, config.NUM_CLASSES - 1))

    # ---- OOD gate ------------------------------------------------------
    # Feed the *un-masked* frame: the fundus-likelihood cues include the round
    # camera aperture and the dark surround, which preprocessing deliberately
    # replaces with flat grey. (ood.py detects padding and neutralises those two
    # cues if it does get a masked image, so this is a quality choice, not a
    # correctness one.)
    ood_input = original_rgb if isinstance(original_rgb, np.ndarray) else raw
    try:
        ood_result = ood.check(ood_input, features=feats)
    except Exception as exc:
        ood_result = {"is_ood": False, "checked": False, "level": "unknown",
                      "reason": f"OOD check unavailable ({exc}).",
                      "reasons": [], "likelihood": None, "cues": {},
                      "weakest_cue": None, "feature_space": None,
                      "feature_check_available": False}

    # ---- Abstention decision -------------------------------------------
    try:
        decision = uncertainty.decide(unc, quality_score=quality_score,
                                      ood=ood_result)
    except Exception:
        decision = {"verdict": "confident", "report_grade": True,
                    "effective_grade": grade, "escalated": False,
                    "reasons": [], "action": "", "n_hard": 0, "n_soft": 0}

    # The decision may escalate a near-tie that straddles the referral threshold
    # to the more urgent grade. That escalated value - not the bare argmax - is
    # what the patient should be managed on, so it becomes the reported grade.
    # The raw argmax is kept as `model_grade` so nothing is hidden.
    model_grade = grade
    acted_grade = int(np.clip(decision.get("effective_grade", grade),
                              0, config.NUM_CLASSES - 1))
    escalated = bool(decision.get("escalated")) and acted_grade != model_grade

    # ---- Explanation ---------------------------------------------------
    if model.mode == "trained":
        try:
            explanation = model.gradcam(enhanced, acted_grade)
            explanation_kind = "Grad-CAM - regions driving the model's decision"
        except Exception:
            explanation = lesion["overlay"] if lesion else raw
            explanation_kind = "Lesion overlay (classical CV)"
    else:
        explanation = lesion["overlay"] if lesion is not None else raw
        explanation_kind = "Lesion overlay (classical CV)"

    # ---- Other conditions from the same photo --------------------------
    others = None
    if screen_other_conditions:
        try:
            # Cataract and defocus look identical to a camera, so the media-opacity
            # test needs to know whether the capture was actually in focus. That is
            # a Laplacian variance (`quality["sharpness"]`), not the blended 0-100
            # quality score. With no sharpness measurement available the safe
            # default is False, i.e. "recapture before blaming an opacity".
            sharp_ok = (sharpness is not None
                        and float(sharpness) >= config.CLARITY_TRUST_SHARPNESS)
            others = multi_disease.screen(raw, capture_sharp_ok=sharp_ok)
        except Exception:
            others = None

    return {
        "mode": model.mode,
        "source": model.source,
        "probs": [float(p) for p in probs],
        "grade": acted_grade,
        "label": config.CLASS_NAMES[acted_grade],
        "confidence": float(probs[acted_grade]),
        # What the network/grader actually argmaxed, before any safety escalation.
        "model_grade": model_grade,
        "model_label": config.CLASS_NAMES[model_grade],
        "escalated": escalated,
        "enhanced": enhanced,
        "raw": raw,
        "explanation": explanation,
        "explanation_kind": explanation_kind,
        "lesion": lesion,
        # new
        "uncertainty": unc,
        "decision": decision,
        "ood": ood_result,
        "multi_disease": others,
        "view_probs": np.asarray(view_probs).tolist(),
    }


def referral_for(grade: int) -> dict:
    title, detail, urgency = config.REFERRAL[int(np.clip(grade, 0, 4))]
    return {"title": title, "detail": detail, "urgency": urgency}
