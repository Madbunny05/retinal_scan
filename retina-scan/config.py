"""
Central configuration for RetinaScan.

Everything a user might want to tweak lives here so the rest of the code
stays clean. All values have safe defaults.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Place a trained checkpoint here for real, fully-offline DR grading.
# Accepts either a raw state_dict (.pth) or a dict with a "state_dict" key.
LOCAL_CHECKPOINT = MODELS_DIR / "dr_model.pth"

# Feature-space OOD reference stats, written by train.py --fit-ood.
# Absent => the Mahalanobis OOD test is skipped (classical check still runs).
OOD_STATS_PATH = MODELS_DIR / "ood_stats.npz"

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
# timm backbone name. efficientnet_b0 is small + fast (good on a laptop CPU).
# For higher accuracy try "tf_efficientnet_b3" or "efficientnet_b3".
BACKBONE = "efficientnet_b0"

# Input resolution fed to the network (square). 384 keeps small DR lesions
# visible; drop to 224 if you need faster CPU inference.
IMG_SIZE = 384

# Optional: a HuggingFace Hub repo that hosts a compatible checkpoint.
# Leave as None unless you have verified a repo that matches BACKBONE + 5 classes.
# Example: HF_REPO = "your-username/dr-efficientnet-b0"; HF_FILE = "dr_model.pth"
HF_REPO = None
HF_FILE = "dr_model.pth"

# --------------------------------------------------------------------------
# Clinical grading  (International Clinical DR Severity Scale)
# --------------------------------------------------------------------------
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
NUM_CLASSES = len(CLASS_NAMES)

# Colour per grade (used in the UI badges + report).
GRADE_COLORS = {
    0: "#1a9850",  # green
    1: "#a6d96a",  # light green
    2: "#fee08b",  # amber
    3: "#fc8d59",  # orange
    4: "#d73027",  # red
}

# Referral guidance keyed by predicted grade.
REFERRAL = {
    0: ("Routine screening", "Re-screen in 12 months.", "low"),
    1: ("Routine monitoring", "Re-screen in 6-12 months.", "low"),
    2: ("Refer to ophthalmologist", "Specialist review within a few weeks.", "moderate"),
    3: ("Refer soon", "Specialist review within days to 2 weeks.", "high"),
    4: ("Urgent referral", "Prompt specialist review; risk of vision loss.", "urgent"),
}

# ImageNet normalisation (matches timm pretrained backbones).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Quality-check thresholds (tunable).
QC_MIN_LAPLACIAN_VAR = 45.0     # below this => likely blurry
QC_MIN_BRIGHTNESS = 18.0        # mean grey below this => too dark
QC_MAX_BRIGHTNESS = 235.0       # mean grey above this => washed out / overexposed
QC_MIN_FOV_RATIO = 0.10         # fraction of frame that must be "retina" coloured

# --------------------------------------------------------------------------
# Uncertainty, calibration and abstention
# --------------------------------------------------------------------------
# Test-time augmentation: how many label-preserving views to average over.
# More views = steadier estimate + a stronger stability signal, but linearly
# more compute. 4 is a good CPU/quality trade-off.
TTA_VIEWS = 4

# MC-dropout: keep dropout active at inference and sample this many times to
# gauge the model's *own* uncertainty (as opposed to a genuinely borderline
# image). Set to 0 to disable. Only applies when the backbone has dropout.
MC_DROPOUT_SAMPLES = 8

# Temperature scaling. Softmax from a cross-entropy-trained net is
# systematically overconfident; T > 1 flattens it without changing the argmax.
# Learn this on a validation set (train.py stores it in the checkpoint) —
# 1.0 means "uncalibrated", and the UI says so rather than pretending.
DEFAULT_TEMPERATURE = 1.0

# --- Abstention thresholds (hard: refuse to report a grade) ---------------
ABSTAIN_MIN_CONFIDENCE = 0.40    # top-class probability floor
ABSTAIN_MIN_MARGIN = 0.08        # gap between top-1 and top-2
ABSTAIN_MAX_ENTROPY = 0.80       # normalised entropy ceiling (0-1)
ABSTAIN_MIN_TTA_AGREEMENT = 0.50 # fraction of augmented views agreeing
ABSTAIN_MAX_MC_STD = 0.22        # std of top-class prob across dropout passes
ABSTAIN_MIN_QUALITY = 30         # capture quality score floor (0-100)

# --- Review thresholds (soft: report, but flag for human review) ----------
REVIEW_MIN_CONFIDENCE = 0.60
REVIEW_MIN_MARGIN = 0.20
REVIEW_MAX_ENTROPY = 0.60
REVIEW_MIN_TTA_AGREEMENT = 0.75

# Grade at and above which the patient must be referred to an ophthalmologist.
# A near-tie *within* one side of this line (e.g. 3 vs 4) does not change what
# happens to the patient. A near-tie *across* it (1 vs 2) does — so that case
# escalates to the more urgent grade instead of being reported as a coin flip.
REFERRAL_THRESHOLD_GRADE = 2

# --------------------------------------------------------------------------
# Out-of-distribution rejection
# --------------------------------------------------------------------------
# Classical "does this look like a retina at all" score (0-1).
OOD_MIN_FUNDUS_LIKELIHOOD = 0.35   # below => refuse to grade
OOD_WARN_FUNDUS_LIKELIHOOD = 0.55  # below => atypical capture, warn

# Feature-space test: flag when the Mahalanobis distance exceeds this multiple
# of the in-distribution reference percentile stored in ood_stats.npz.
OOD_MAHALANOBIS_RATIO = 2.5

# --------------------------------------------------------------------------
# Multi-condition screening (classical morphometry, not trained models)
# --------------------------------------------------------------------------
# Glaucoma: vertical cup-to-disc ratio. 0.6+ is a common referral threshold.
GLAUCOMA_VCDR_BORDERLINE = 0.55
GLAUCOMA_VCDR_REFER = 0.65

# Hypertensive retinopathy: arteriole-to-venule ratio. Healthy is ~0.66;
# chronic hypertension narrows arterioles and drives it down.
AVR_BORDERLINE = 0.62
AVR_REFER = 0.55
TORTUOSITY_FLAG = 1.35

# Cataract / media opacity: clarity score 0-100 (higher = clearer).
CLARITY_BORDERLINE = 55
CLARITY_REFER = 35

# Sharpness above which we trust that low clarity is real opacity, not shake.
CLARITY_TRUST_SHARPNESS = 60.0

APP_TITLE = "RetinaScan"
APP_TAGLINE = "Offline smartphone diabetic retinopathy screening"
DISCLAIMER = (
    "RetinaScan is a screening aid, not a diagnostic device. Results are "
    "probabilistic and must be confirmed by a qualified clinician. Do not use "
    "as the sole basis for any medical decision."
)
