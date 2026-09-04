"""
RetinaScan - offline smartphone diabetic retinopathy screening.

Run:      streamlit run app.py
Phone:    open http://<your-laptop-ip>:8501 on a phone on the same Wi-Fi
          (the camera widget works right in the mobile browser)

Everything here runs locally. No image or result ever leaves the device.
"""
import hashlib

import numpy as np
import streamlit as st

import config
import dr_model
import education
import preprocessing
import report
import uncertainty

st.set_page_config(
    page_title=f"{config.APP_TITLE} - DR screening",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
      :root {--ink:#12251f; --muted:#66756f; --pine:#075e4b; --mint:#eaf7f1;
             --line:#dce8e2; --paper:#ffffff;}
      .stApp {background:linear-gradient(135deg,#f7fbf9 0%,#ffffff 46%,#f3faf7 100%); color:var(--ink);}
      .block-container {padding-top:1.25rem; padding-bottom:3rem; max-width:1180px;}
      [data-testid="stSidebar"] {background:#f5faf7; border-right:1px solid var(--line);}
      [data-testid="stSidebar"] > div:first-child {padding-top:1.25rem;}
      [data-testid="stMetric"] {background:rgba(255,255,255,.78); border:1px solid var(--line);
          border-radius:14px; padding:12px 14px; box-shadow:0 3px 12px rgba(18,37,31,.035);}
      [data-testid="stMetricLabel"] {color:var(--muted); font-size:.78rem; font-weight:650;}
      [data-testid="stMetricValue"] {font-size:1.35rem; color:var(--ink);}
      .rs-hero {padding:27px 30px; border-radius:22px; margin:0 0 18px;
          background:radial-gradient(circle at 88% 0%,#37aa82 0,transparent 27%),linear-gradient(120deg,#063f36,#08705a);
          color:#fff; box-shadow:0 12px 28px rgba(7,94,75,.18);}
      .rs-hero h1 {font-size:2rem; line-height:1.1; margin:0 0 8px; letter-spacing:-.035em;}
      .rs-hero p {margin:0; color:#d7f0e7; font-size:1rem; max-width:680px;}
      .rs-kicker {color:#aee7d1; text-transform:uppercase; letter-spacing:.1em; font-size:.68rem; font-weight:750; margin-bottom:8px;}
      .rs-badge {display:inline-block; padding:4px 10px; border-radius:999px; margin:12px 5px 0 0;
                 font-size:.72rem; font-weight:650; background:rgba(255,255,255,.12); color:#effcf7;
                 border:1px solid rgba(255,255,255,.23);}
      .rs-grade {border-radius:18px; padding:20px 22px; color:#fff; margin-bottom:12px; box-shadow:0 8px 20px rgba(18,37,31,.12);}
      .rs-grade h1 {margin:0; font-size:2.1rem; line-height:1.1;}
      .rs-grade p {margin:2px 0 0; opacity:.9; font-size:.95rem;}
      .rs-bar-track {background:#edf3f0; border-radius:7px; height:23px; width:100%;
                     position:relative; margin:5px 0; overflow:hidden;}
      .rs-bar-fill {height:22px; border-radius:6px;}
      .rs-bar-label {position:absolute; left:8px; top:1px; font-size:.8rem; color:#111;}
      .rs-bar-val {position:absolute; right:8px; top:1px; font-size:.8rem; color:#111;}
      .rs-ref {border-radius:14px; padding:15px 17px; border:1px solid var(--line); background:#fff;}
      .rs-disc {color:#7a828a; font-size:0.78rem; border-top:1px solid #eef1f4;
                padding-top:10px; margin-top:14px;}
      .rs-edu {border:1px solid var(--line); border-radius:14px; padding:16px 19px; background:#fff;}
      .rs-edu h4 {margin:0 0 6px; font-size:1rem;}
      .rs-panel {border-radius:14px; padding:14px 17px; height:100%;}
      .rs-avoid {background:#fdecea; border:1px solid #f5c6c0;}
      .rs-favor {background:#e7f5ee; border:1px solid #b7e0cd;}
      .rs-panel h4 {margin:0 0 8px; font-size:0.95rem;}
      .rs-panel ul {margin:0; padding-left:18px;}
      .rs-panel li {margin:4px 0; font-size:0.88rem; line-height:1.35;}
      .rs-verdict {border-radius:14px; padding:15px 18px; margin-bottom:14px; box-shadow:0 3px 10px rgba(18,37,31,.025);}
      .rs-verdict h4 {margin:0 0 6px; font-size:1.05rem;}
      .rs-verdict ul {margin:6px 0 0; padding-left:20px;}
      .rs-verdict li {font-size:0.86rem; margin:3px 0; line-height:1.35;}
      .rs-find {border:1px solid var(--line); border-left-width:6px;
                border-radius:12px; padding:12px 15px; margin-bottom:9px; background:#fff;}
      .rs-find b {font-size:0.92rem;}
      .rs-find p {margin:4px 0 0; font-size:0.84rem; color:#3d4650; line-height:1.4;}
      .rs-metric-chip {display:inline-block; background:#f4f6f8; border:1px solid #e3e7eb;
                       border-radius:6px; padding:1px 7px; font-size:0.74rem;
                       color:#4a5560; margin:4px 4px 0 0;}
      .rs-empty {background:#fff; border:1px solid var(--line); border-radius:18px; padding:20px 22px;
          margin-top:14px; box-shadow:0 6px 18px rgba(18,37,31,.04);}
      .rs-empty h3 {margin:0 0 5px; color:var(--ink); font-size:1.1rem;}
      .rs-empty p {margin:0; color:var(--muted); font-size:.9rem;}
      [data-testid="stFileUploader"], [data-testid="stCameraInput"] {border-radius:14px; overflow:hidden;}
      .stTabs [data-baseweb="tab-list"] {gap:8px; border-bottom:1px solid var(--line);}
      .stTabs [data-baseweb="tab"] {height:42px; border-radius:9px 9px 0 0; padding:0 16px; font-weight:600; color:var(--muted);}
      .stTabs [aria-selected="true"] {color:var(--pine); background:#eaf7f1;}
      .stButton button, .stDownloadButton button {border-radius:10px; font-weight:650;}
      @media (max-width:700px) { .block-container {padding:1rem .8rem 2rem;} .rs-hero {padding:22px 20px; border-radius:17px;} .rs-hero h1 {font-size:1.6rem;} [data-testid="stMetric"] {padding:10px;} }
    </style>
    """,
    unsafe_allow_html=True,
)

URGENCY_COLOR = {"low": "#1a9850", "moderate": "#f0ad4e",
                 "high": "#fc8d59", "urgent": "#d73027"}


def prob_bars_html(probs) -> str:
    rows = []
    for i, p in enumerate(probs):
        color = config.GRADE_COLORS[i]
        pct = max(2.0, p * 100)
        rows.append(
            f"<div class='rs-bar-track'>"
            f"<div class='rs-bar-fill' style='width:{pct:.1f}%;background:{color}'></div>"
            f"<span class='rs-bar-label'>{i} · {config.CLASS_NAMES[i]}</span>"
            f"<span class='rs-bar-val'>{p*100:.1f}%</span>"
            f"</div>"
        )
    return "".join(rows)


def _li(items) -> str:
    return "".join(f"<li>{x}</li>" for x in items)


@st.cache_resource(show_spinner=False)
def _load_model_status():
    m = dr_model.get_model()
    return m.mode, m.source


def analyze(img_bytes: bytes, enhance: bool, use_tta: bool, screen_others: bool):
    key = hashlib.md5(img_bytes).hexdigest() + f"{enhance}{use_tta}{screen_others}"
    if st.session_state.get("_key") != key:
        rgb = preprocessing.load_rgb(img_bytes)
        with st.spinner("Analysing fundus image…"):
            # Quality first: the abstention logic needs it as an input signal.
            quality = preprocessing.quality_check(img_bytes)
            result = dr_model.run_inference(
                rgb, enhance=enhance, use_tta=use_tta,
                quality_score=quality["score"],
                sharpness=quality.get("sharpness"),
                screen_other_conditions=screen_others,
            )
        st.session_state["_key"] = key
        st.session_state["_result"] = result
        st.session_state["_quality"] = quality
    return st.session_state["_result"], st.session_state["_quality"]


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
mode, source = _load_model_status()

with st.sidebar:
    st.markdown(f"### {config.APP_TITLE}")
    st.caption(config.APP_TAGLINE)
    if mode == "trained":
        st.success(f"Model: trained CNN\n\n{source}")
    else:
        st.warning(
            "Model: **DEMO mode** (classical CV lesion grader).\n\n"
            "No trained checkpoint found. Drop a checkpoint at "
            "`models/dr_model.pth` (or train one with `train.py`) for real "
            "CNN grading. See README."
        )
    st.divider()
    st.markdown("**Settings**")
    enhance = st.toggle("Contrast enhancement (Ben-Graham)", value=True,
                        help="Boosts micro-lesion visibility before inference.")
    use_tta = st.toggle(
        f"Uncertainty estimation ({config.TTA_VIEWS}-view TTA)", value=True,
        help="Predicts over flipped/rotated copies of the same eye. Views that "
             "disagree signal an unstable prediction, which triggers abstention "
             "instead of a confident guess. Costs ~4x inference time.")
    screen_others = st.toggle(
        "Screen for other conditions", value=True,
        help="Also estimates glaucoma (cup-to-disc ratio), hypertensive "
             "retinopathy (arteriole-to-venule ratio) and media clarity "
             "(cataract) from the same photo.")
    st.divider()
    st.markdown("**Patient (optional, for report)**")
    p_id = st.text_input("Patient ID")
    p_name = st.text_input("Name")
    p_eye = st.selectbox("Eye", ["", "Right (OD)", "Left (OS)"])
    p_notes = st.text_area("Notes", height=70)
    st.divider()
    st.caption("🔒 Runs fully on-device · no internet required")


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    f"<section class='rs-hero'><div class='rs-kicker'>Offline retinal screening</div>"
    f"<h1>Clearer insight from a single fundus image.</h1>"
    f"<p>{config.APP_TAGLINE}</p>"
    f"<span class='rs-badge'>● Fully on-device</span> "
    f"<span class='rs-badge'>5-class ICDR grading</span> "
    f"<span class='rs-badge'>Safety-first uncertainty checks</span> "
    f"<span class='rs-badge'>4 conditions · 1 photo</span></section>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
st.markdown("#### Start a screening")
st.caption("Use a well-lit, centred retinal fundus image. Your image is analysed locally and is never uploaded.")
tab_cam, tab_up = st.tabs(["📷 Capture with camera", "📁 Upload from device"])
img_bytes = None
with tab_cam:
    st.caption("Attach a fundus lens adapter to the phone camera, centre the "
               "optic disc, then capture.")
    shot = st.camera_input("Capture fundus", label_visibility="collapsed")
    if shot is not None:
        img_bytes = shot.getvalue()
with tab_up:
    up = st.file_uploader("Upload a fundus photo",
                          type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                          label_visibility="collapsed")
    if up is not None:
        img_bytes = up.getvalue()

if img_bytes is None:
    st.markdown(
        "<div class='rs-empty'><h3>Ready when you are</h3>"
        "<p>Capture a fundus image or choose one from your device to receive a "
        "quality check, severity screen, referral guidance, and a shareable report.</p></div>",
        unsafe_allow_html=True,
    )
    st.info("Tip: for phone capture, centre the optic disc and keep the lens adapter clean and steady.")
    st.markdown(f"<div class='rs-disc'>{config.DISCLAIMER}</div>", unsafe_allow_html=True)
    st.stop()

# --------------------------------------------------------------------------
# Analyse
# --------------------------------------------------------------------------
try:
    result, quality = analyze(img_bytes, enhance, use_tta, screen_others)
except Exception as exc:  # never hard-crash the demo
    st.error(f"Could not process this image: {exc}")
    st.stop()

grade = result["grade"]
ref = dr_model.referral_for(grade)
grade_color = config.GRADE_COLORS[grade]
unc = result.get("uncertainty", {})
decision = result.get("decision", {"verdict": "confident", "report_grade": True,
                                   "reasons": [], "action": ""})
ood_info = result.get("ood", {})
verdict = decision.get("verdict", "confident")
style = uncertainty.VERDICT_STYLE.get(verdict, uncertainty.VERDICT_STYLE["confident"])

# --------------------------------------------------------------------------
# Reliability verdict  -  shown BEFORE the grade, because it decides whether
# a grade should be trusted at all.
# --------------------------------------------------------------------------
bg = {"confident": "#e7f5ee", "borderline": "#fff6e5",
      "inconclusive": "#fdecea"}.get(verdict, "#f4f6f8")
brd = {"confident": "#b7e0cd", "borderline": "#f3d9a4",
       "inconclusive": "#f5c6c0"}.get(verdict, "#e3e7eb")
head = {
    "confident": "Reliable read",
    "borderline": "Provisional read — flag for human review",
    "inconclusive": "Inconclusive — no grade reported",
}.get(verdict, "Reliability unknown")
if result.get("escalated"):
    head = "Escalated for safety — flag for human review"
st.markdown(
    f"<div class='rs-verdict' style='background:{bg};border:1px solid {brd};"
    f"border-left:6px solid {style['color']}'>"
    f"<h4 style='color:{style['color']}'>{style['icon']} {head}</h4>"
    f"<div style='font-size:0.88rem'>{decision.get('action','')}</div>"
    f"<ul>{_li(decision.get('reasons', []))}</ul></div>",
    unsafe_allow_html=True,
)

# Quality strip
if quality["warnings"]:
    st.warning("Image quality notes: " + "  •  ".join(quality["warnings"]))

# Top metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Predicted grade",
          "—" if not decision.get("report_grade", True)
          else f"{grade} · {config.CLASS_NAMES[grade]}",
          delta=(f"raised from {result.get('model_grade')}"
                 if result.get("escalated") else None),
          delta_color="off",
          help="The model's two top grades were nearly tied on opposite sides "
               "of the referral threshold, so the more urgent one is reported. "
               "A screening tool should err towards referral."
               if result.get("escalated") else None)
m2.metric("Confidence", f"{result['confidence']*100:.0f}%",
          help="Calibrated top-class probability."
               if unc.get("calibrated") else
               "Raw softmax probability — no validation set was available to "
               "calibrate it, so treat it as a ranking, not a true likelihood.")
m3.metric("View agreement", f"{unc.get('tta_agreement', 1.0)*100:.0f}%",
          help=f"Share of the {unc.get('tta_views', 1)} flipped/rotated views "
               f"that produced the same grade. Low agreement means the "
               f"prediction is unstable.")
m4.metric("Capture quality", f"{quality['score']}/100")
m5.metric("Engine", "Trained CNN" if result["mode"] == "trained" else "Demo (CV)")

# --------------------------------------------------------------------------
# Hard stop: an out-of-distribution or unusable image gets no grade at all.
# --------------------------------------------------------------------------
if not decision.get("report_grade", True):
    st.write("")
    c_img, c_expl = st.columns([1, 1.15])
    with c_img:
        st.image(result["enhanced"], caption="Processed capture",
                 use_container_width=True)
    with c_expl:
        st.markdown(
            "#### Why no grade was reported\n"
            "RetinaScan deliberately refuses to output a severity grade it "
            "cannot stand behind. A confidently wrong screening result is worse "
            "than no result: it can send a patient with proliferative disease "
            "home for twelve months.\n\n"
            "**What to do now**"
        )
        st.markdown(
            "- Retake the photo: steady the phone, improve the illumination, "
            "and centre the optic disc in the frame.\n"
            "- Check the fundus lens adapter is seated and clean.\n"
            "- If repeated good captures still come back inconclusive, refer "
            "the patient for a manual dilated examination."
        )
    with st.expander("Technical detail — uncertainty and distribution checks"):
        st.caption(uncertainty.summary_line(unc))
        if ood_info.get("checked"):
            st.markdown(f"**Fundus-likelihood:** {ood_info.get('likelihood')} "
                        f"(reject below {config.OOD_MIN_FUNDUS_LIKELIHOOD})")
            if ood_info.get("cues"):
                st.markdown(" ".join(
                    f"<span class='rs-metric-chip'>{k.replace('_',' ')}: {v}</span>"
                    for k, v in ood_info["cues"].items()), unsafe_allow_html=True)
            fs = ood_info.get("feature_space")
            if fs:
                # `ratio` is None when ood_stats.npz carries no reference distance,
                # in which case a bare distance is all we can honestly show.
                if fs.get("ratio") is not None:
                    st.markdown(f"**Feature-space Mahalanobis:** {fs['distance']} "
                                f"({fs['ratio']}× in-distribution reference)")
                else:
                    st.markdown(f"**Feature-space Mahalanobis:** {fs['distance']} "
                                f"(no in-distribution reference stored, so this "
                                f"number has no scale yet)")
            else:
                st.caption("Feature-space OOD test unavailable (no "
                           "`models/ood_stats.npz` — see README).")
        st.markdown("**Per-grade probabilities (for transparency only)**")
        st.markdown(prob_bars_html(result["probs"]), unsafe_allow_html=True)
    st.markdown(f"<div class='rs-disc'>{config.DISCLAIMER}</div>",
                unsafe_allow_html=True)
    st.stop()

st.write("")
left, right = st.columns([1.05, 1])

with left:
    st.markdown(
        f"<div class='rs-grade' style='background:{grade_color}'>"
        f"<h1>Grade {grade}</h1><p>{config.CLASS_NAMES[grade]} · "
        f"{result['confidence']*100:.0f}% confidence</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("**Severity probabilities**")
    st.markdown(prob_bars_html(result["probs"]), unsafe_allow_html=True)

    uc = URGENCY_COLOR.get(ref["urgency"], "#555")
    st.markdown(
        f"<div class='rs-ref' style='border-left:6px solid {uc}'>"
        f"<b>{ref['title']}</b> "
        f"<span style='color:{uc};font-weight:600'>({ref['urgency'].upper()})</span><br>"
        f"{ref['detail']}</div>",
        unsafe_allow_html=True,
    )

    lesion = result.get("lesion")
    if lesion:
        st.caption(
            f"Classical CV findings — red lesions: **{lesion['red_lesion_count']}**, "
            f"exudate area: **{lesion['exudate_area_frac']*100:.2f}%** of FOV"
        )

with right:
    c1, c2 = st.columns(2)
    with c1:
        st.image(result["enhanced"], caption="Processed fundus",
                 use_container_width=True)
    with c2:
        st.image(result["explanation"], caption=result["explanation_kind"],
                 use_container_width=True)

st.write("")

# --------------------------------------------------------------------------
# How sure is the model?  (uncertainty breakdown)
# --------------------------------------------------------------------------
with st.expander("🎯 How sure is this result? — uncertainty breakdown"):
    st.caption(
        "A screening model that is confidently wrong is more dangerous than one "
        "that admits doubt. These are the signals RetinaScan checks before it "
        "agrees to report a grade."
    )
    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Top-2 margin", f"{unc.get('margin', 0)*100:.0f} pts",
              help="Gap between the winning grade and the runner-up. A narrow "
                   "gap means the image sits on a decision boundary.")
    u2.metric("Entropy", f"{unc.get('entropy_norm', 0):.2f}",
              help="0 = all probability on one grade, 1 = uniform across all "
                   "five. High entropy means the model is spreading its bet.")
    u3.metric("Expected grade", f"{unc.get('expected_grade', grade):.2f}",
              help="Probability-weighted grade. DR severity is ordinal, so this "
                   "is often more informative than the argmax — an expected "
                   "grade of 1.5 means the image genuinely straddles two stages.")
    mc_std = unc.get("mc_dropout_std")
    u4.metric("MC-dropout σ", "n/a" if mc_std is None else f"{mc_std:.3f}",
              help="Spread of the top-class probability across stochastic "
                   "forward passes with dropout left on. Available only for "
                   "trained CNNs that contain dropout layers.")

    st.markdown("**Grade predicted per augmented view**")
    vp = result.get("view_probs") or []
    if vp:
        rows = []
        for i, row in enumerate(vp):
            g = int(np.argmax(row))
            name = ["original", "h-flip", "v-flip", "180°", "90°", "270°"][i] \
                if i < 6 else f"view {i}"
            rows.append(
                f"<span class='rs-metric-chip'>{name}: grade {g} "
                f"({max(row)*100:.0f}%)</span>")
        st.markdown(" ".join(rows), unsafe_allow_html=True)
        st.caption(
            "A flipped retina is still the same retina, so a clinician would "
            "grade every one of these identically. Disagreement here is pure "
            "model instability."
        )
    else:
        st.caption("Test-time augmentation is switched off in the sidebar.")

    st.markdown("**Distribution check**")
    if ood_info.get("checked"):
        st.markdown(
            f"Fundus-likelihood **{ood_info.get('likelihood')}** "
            f"({ood_info.get('level', 'n/a')}) — rejects below "
            f"{config.OOD_MIN_FUNDUS_LIKELIHOOD}, warns below "
            f"{config.OOD_WARN_FUNDUS_LIKELIHOOD}.")
        if ood_info.get("cues"):
            st.markdown(" ".join(
                f"<span class='rs-metric-chip'>{k.replace('_',' ')}: {v}</span>"
                for k, v in ood_info["cues"].items()), unsafe_allow_html=True)
        fs = ood_info.get("feature_space")
        if fs:
            if fs.get("ratio") is not None:
                st.caption(f"Feature-space Mahalanobis {fs['distance']} "
                           f"({fs['ratio']}× the in-distribution reference).")
            else:
                st.caption(f"Feature-space Mahalanobis {fs['distance']} — no "
                           f"in-distribution reference was stored, so there is "
                           f"nothing to compare it against yet.")
        else:
            st.caption(
                "Feature-space (Mahalanobis) OOD test not active — it needs "
                "`models/ood_stats.npz`, written by `train.py --fit-ood`. The "
                "classical check above runs regardless.")
    st.caption(uncertainty.summary_line(unc))

# --------------------------------------------------------------------------
# Other conditions screened from the same photo
# --------------------------------------------------------------------------
others = result.get("multi_disease")
if others:
    n_ref = others["n_referrals"]
    n_bl = others["n_borderline"]
    if n_ref:
        summary = f"— **{n_ref} finding(s) needing referral**"
    elif n_bl:
        summary = f"— {n_bl} borderline finding(s)"
    else:
        summary = "— nothing else of concern"
    st.markdown(f"### Other conditions screened from this same photo {summary}")
    st.caption(
        "One capture, four screens. In a camp where the patient may never see "
        "an ophthalmologist again, the DR photo already contains signal for "
        "three other leading causes of avoidable blindness."
    )

    f_col, i_col = st.columns([1.35, 1])
    with f_col:
        for f in others["findings"]:
            col = {"refer": "#d73027", "borderline": "#f0ad4e",
                   "normal": "#1a9850", "unavailable": "#9aa4ae"}.get(
                       f.get("status", ""), "#9aa4ae")
            chips = " ".join(
                f"<span class='rs-metric-chip'>{k}: {v}</span>"
                for k, v in (f.get("metrics") or {}).items())
            st.markdown(
                f"<div class='rs-find' style='border-left-color:{col}'>"
                f"<b>{f['condition']}</b> "
                f"<span style='color:{col};font-weight:600;font-size:0.78rem'>"
                f"{f.get('status', 'unknown').upper()}</span><br>"
                f"<b style='font-weight:500'>{f['headline']}</b>"
                f"<p>{f['detail']}</p>{chips}</div>",
                unsafe_allow_html=True,
            )
    with i_col:
        st.image(others["overlay"], caption="Structures measured",
                 use_container_width=True)
        st.caption(others["overlay_legend"])

    st.caption("⚠ " + others["disclaimer"])
    st.write("")

# --------------------------------------------------------------------------
# Understanding your result  (disease explanation · treatment · diet)
# --------------------------------------------------------------------------
edu = education.get_education(grade)
st.markdown("### Understanding your result")
tab_what, tab_treat, tab_diet = st.tabs(
    ["🩺 What it means", "💊 Treatment & management", "🥗 Diet guidance"]
)

with tab_what:
    st.markdown(
        f"<div class='rs-edu'><h4>Grade {grade} · {edu['stage']}</h4>"
        f"{edu['what_it_is']}</div>",
        unsafe_allow_html=True,
    )

with tab_treat:
    st.markdown(
        f"<div class='rs-edu'><h4>How this stage is treated &amp; managed</h4>"
        f"<ul>{_li(edu['treatment'])}</ul></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Diabetic retinopathy usually can't be fully cured, but good care "
        "protects your sight - and catching it early can even reverse the "
        "damage."
    )

with tab_diet:
    st.caption(edu["diet_rationale"])
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            f"<div class='rs-panel rs-avoid'><h4>⚠️ Foods to avoid</h4>"
            f"<ul>{_li(edu['foods_to_avoid'])}</ul></div>",
            unsafe_allow_html=True,
        )
    with d2:
        st.markdown(
            f"<div class='rs-panel rs-favor'><h4>✅ Eye-friendly foods</h4>"
            f"<ul>{_li(edu['foods_to_favor'])}</ul></div>",
            unsafe_allow_html=True,
        )

st.caption(edu["disclaimer"])
st.write("")

# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
patient = {"id": p_id, "name": p_name, "eye": p_eye, "notes": p_notes}
try:
    pdf = report.build_report(result, quality, patient)
    fname = f"retinascan_{p_id or 'patient'}_grade{grade}.pdf"
    st.download_button("⬇ Download PDF report", data=pdf, file_name=fname,
                       mime="application/pdf", type="primary")
except Exception as exc:
    st.caption(f"(PDF report unavailable: {exc})")

if result["mode"] != "trained":
    st.info("You're in **demo mode** (classical CV grader). Add a trained "
            "checkpoint for real CNN predictions — see the README.", icon="ℹ")

st.markdown(f"<div class='rs-disc'>{config.DISCLAIMER}</div>", unsafe_allow_html=True)
