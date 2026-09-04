# 👁 RetinaScan — Offline Smartphone Diabetic Retinopathy Screening

A point-of-care app that grades **diabetic retinopathy (DR)** from a retinal
fundus image — captured on a phone or uploaded — and runs **entirely on-device
with no internet connection**. Built for a hackathon.

It outputs a 5-class severity grade (International Clinical DR Severity Scale),
a confidence score, an explainability overlay, a referral recommendation, and a
downloadable PDF report — all locally.

---

## Why it meets the brief

| Requirement | How RetinaScan delivers |
|---|---|
| **Smartphone-based fundus imaging** | Live capture via the phone browser camera (`st.camera_input`) — point the phone through a clip-on fundus lens adapter — **plus** image upload from the gallery. |
| **Works offline** | All inference is local (a CNN running via PyTorch, or a classical-CV grader). No image or result ever leaves the device. No cloud API is called at inference time. |

---

## Features

- **5-class ICDR grading:** No DR · Mild · Moderate · Severe · Proliferative
- **Real CNN** (EfficientNet via `timm`) when a trained checkpoint is present;
  automatic, clearly-labelled **classical-CV fallback** so a live demo never crashes
- **Knows when to abstain** — quantifies its own uncertainty and refuses to
  report a grade it cannot stand behind (see below)
- **Four conditions from one photo** — diabetic retinopathy plus glaucoma risk,
  hypertensive retinopathy and media opacity / cataract (see below)
- **Out-of-distribution rejection** — detects "that isn't a retina" and
  "that isn't a retina like the ones I was trained on"
- **Fundus preprocessing** (crop-to-disc, circular mask, Ben-Graham contrast enhancement)
- **Capture quality check** — warns on blur, poor lighting, off-centre / non-fundus frames
- **Explainability** — Grad-CAM heatmap (CNN) or lesion overlay highlighting
  likely haemorrhages/micro-aneurysms and exudates
- **Referral guidance** per grade, with urgency level
- **Offline PDF report** with patient details, images, probabilities, the
  reliability verdict, an uncertainty audit and the multi-condition findings
- **Phone-friendly** — open the URL on a phone on the same Wi-Fi

---

## Two things that make this different

### 1. It knows when it doesn't know

Published DR models degrade sharply on cameras and populations they weren't
trained on, and the failure is **silent** — you get a confident number that is
wrong. A screening tool that sends a patient with proliferative disease home for
twelve months does more harm than one that says "retake this photo".

RetinaScan checks four independent signals before it agrees to report anything
(`uncertainty.py`):

| Signal | What it catches |
|---|---|
| **Predictive entropy** | The model is spreading its bet across all five grades. |
| **Top-2 margin** | The image sits on a decision boundary between two grades. |
| **Augmentation agreement (TTA)** | Flips and rotations of the *same eye* — which a clinician would grade identically — disagree. Pure model instability. |
| **MC-dropout variance** | Repeated stochastic passes over the same pixels disagree, so the model's own parameters are unsure. (Trained CNNs with dropout only.) |

Plus two out-of-distribution tests (`ood.py`): a **classical fundus-likelihood**
score built from six retinal cues (red dominance, channel order, circular field,
dark surround, vessel structure, optic disc) that rejects non-fundus frames, and
a **Mahalanobis distance** on penultimate CNN features that flags images far
outside the training distribution — a different camera or population.

These fold into one of three verdicts, shown *above* the grade because it decides
whether the grade should be trusted at all:

- **confident** — report the grade normally
- **borderline** — report it, flagged for human review
- **inconclusive** — report **no grade**; the UI explains why and what to do instead

**The fail-safe detail worth pointing at:** a near-tie is not automatically an
abstention. DR grades are ordinal and what matters is the *referral decision*, so
a coin flip between grades 3 and 4 still tells you everything you need — refer
urgently either way. A coin flip between 1 and 2 straddles the referral threshold
(`REFERRAL_THRESHOLD_GRADE` in `config.py`), so RetinaScan **escalates to the more
urgent grade** rather than reporting nothing. Under-referral is the harmful error;
the tool errs in the safe direction and says so, on screen and in the PDF.

Confidences are honest about calibration: raw softmax from a cross-entropy-trained
net is systematically overconfident, so `train.py` fits a **temperature-scaling**
scalar on the held-out validation split and stores it in the checkpoint. With no
calibration available the UI labels the number "uncalibrated — treat it as a
ranking, not a likelihood" instead of pretending.

### 2. Four screens from one capture

In a camp where the patient may never see an ophthalmologist again, the DR photo
already contains signal for three other leading causes of avoidable blindness.
`multi_disease.py` measures them with classical morphometry (no extra model, no
extra capture, no extra time):

| Condition | Measurement | Referral threshold |
|---|---|---|
| **Glaucoma risk** | Vertical cup-to-disc ratio (VCDR) — disc located from the blurred green channel, cup as the brightest core within it | VCDR ≥ 0.65 |
| **Hypertensive retinopathy** | Arteriole-to-venule ratio (AVR) from vessel calibre in a peripapillary annulus, plus tortuosity | AVR ≤ 0.55 |
| **Cataract / media opacity** | Clarity score from Laplacian variance, saturation and contrast spread | clarity ≤ 35 |

An overlay shows exactly which structures were measured, and every finding is
labelled as classical morphometry rather than a trained model. Low clarity is
deliberately distinguished from camera shake — defocus and cataract look alike,
so a blurry capture is reported as "recapture first", not as opacity.

---

## Quick start

```bash
cd retina-scan
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python selftest.py          # optional: verifies the whole pipeline end-to-end
streamlit run app.py
```

Open http://localhost:8501. Use the **Upload** tab with any fundus image, or the
**Camera** tab. No dataset? Run `python make_sample.py` to generate a synthetic
`sample_fundus.png` you can upload to see everything working.

> **First run & offline:** if you use a trained CNN with `pretrained` backbone
> caching, the very first launch may download backbone weights to your local
> cache. After that (or if you use a local `models/dr_model.pth`), the app runs
> **100% offline**. Inference itself never uses the network.

---

## Demoing on a phone

Find your laptop's LAN IP (e.g. `ipconfig` / `ifconfig`) and open
`http://<laptop-ip>:8501` on a phone on the **same Wi-Fi**. Server config already
binds to `0.0.0.0`.

- **Upload tab:** works immediately over plain `http://`.
- **Camera tab:** browsers only allow camera access over `https://` or
  `localhost`. Two offline-friendly options:
  1. **Self-signed HTTPS (recommended).** Generate a cert and enable it — see the
     commented lines in `.streamlit/config.toml`:
     ```bash
     openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 365 -subj "/CN=retinascan"
     ```
     Uncomment `sslCertFile`/`sslKeyFile`, restart, open `https://<laptop-ip>:8501`,
     accept the warning. Camera now works, still fully offline.
  2. **Chrome insecure-origin flag (Android):** `chrome://flags/#unsafely-treat-insecure-origin-as-secure`
     → add `http://<laptop-ip>:8501` → relaunch.
- **On the laptop itself,** the Camera tab works out of the box at `localhost`
  (great if you have a USB fundus camera / webcam).

---

## Getting a real model (three options)

The app looks for weights in this order and shows the active engine in the sidebar.

1. **Train your own (APTOS 2019)** — most authentic for the demo:
   ```bash
   python train.py --data-dir path/to/train_images --csv path/to/train.csv --epochs 8
   ```
   Writes `models/dr_model.pth`. Works with any `image,label` CSV too (see `--help`).

   After the last epoch `train.py` also uses the held-out validation split to:
   - fit the **temperature-scaling** scalar and re-save the checkpoint as a dict
     (`state_dict` + `temperature` + `backbone` + `img_size` + `val_qwk`), so the
     app reports calibrated confidence automatically — skip with `--no-calibrate`;
   - write **`models/ood_stats.npz`** (feature mean + shrunk inverse covariance +
     reference distance) which switches on the Mahalanobis out-of-distribution
     test — skip with `--no-fit-ood`.

   Neither artefact is required. Without them the app still runs; it just labels
   confidence as uncalibrated and falls back to the classical OOD check only.
2. **Drop in a checkpoint** you already have at `models/dr_model.pth`
   (5-class, matching `BACKBONE` in `config.py`).
3. **Pull from Hugging Face** — set `HF_REPO`/`HF_FILE` in `config.py` to a repo
   you've verified hosts a compatible checkpoint.

If none is found, the app runs in **demo mode** (classical CV grader) — real
image-based lesion detection, deterministic output, clearly labelled in the UI.

---

## How grading maps to action (ICDR)

| Grade | Class | Recommendation |
|---|---|---|
| 0 | No DR | Routine re-screen in 12 months |
| 1 | Mild | Monitor, re-screen 6–12 months |
| 2 | Moderate | Refer to ophthalmologist (weeks) |
| 3 | Severe | Refer soon (days–2 weeks) |
| 4 | Proliferative DR | **Urgent** referral |

Tune thresholds and text in `config.py`.

---

## Project layout

```
retina-scan/
├── app.py              # Streamlit UI (camera + upload, results, report)
├── config.py           # all tunables: backbone, image size, classes, referral,
│                       #   abstention thresholds, OOD + multi-condition limits
├── preprocessing.py    # load / crop / mask / Ben-Graham enhance + quality check
├── dr_model.py         # model load, CNN inference, TTA, MC-dropout, Grad-CAM,
│                       #   features(), run_inference()
├── uncertainty.py      # TTA views, temperature scaling, uncertainty metrics,
│                       #   abstain / escalate decision
├── ood.py              # fundus-likelihood cues + Mahalanobis feature-space OOD
├── multi_disease.py    # glaucoma (VCDR), hypertensive (AVR), media clarity
├── lesion_analysis.py  # classical-CV lesion detection + deterministic fallback grader
├── education.py        # per-grade explanation, treatment and diet guidance
├── report.py           # offline PDF report (reportlab)
├── train.py            # optional: train EfficientNet on APTOS 2019 (+ calibrate, + fit OOD)
├── make_sample.py      # synthetic fundus image for quick testing
├── selftest.py         # end-to-end smoke test
├── requirements.txt
├── .streamlit/config.toml
└── models/             # dr_model.pth  and  ood_stats.npz  live here
```

Every new stage is individually failure-tolerant: if uncertainty quantification,
OOD checking or multi-condition screening throws, the core grade still comes back
and the affected panel reports itself as unavailable. A live demo cannot be
crashed by them.

---

## Talking points for judges

- **Access:** turns a $10 clip-on lens + any phone into a DR screening tool for
  clinics with no reliable internet.
- **Safety, not just accuracy:** the model abstains rather than guessing, and when
  a near-tie straddles the referral threshold it escalates to the more urgent
  grade. Most submissions optimise the accuracy number; this one is engineered
  around the *cost asymmetry* of a missed referral.
- **Honest confidence:** temperature-scaled when a validation split is available,
  and explicitly labelled "uncalibrated" when it isn't — no fake likelihoods.
- **Knows what it wasn't trained on:** rejects non-fundus frames outright and
  flags unfamiliar cameras/populations via feature-space Mahalanobis distance.
- **Four diseases, one capture, zero extra hardware:** DR grading plus glaucoma,
  hypertensive retinopathy and cataract screening from the same photo — the
  marginal cost of the extra three screens is a few milliseconds.
- **Trust:** explainable (Grad-CAM / lesion overlay + a measured-structures
  overlay), honest about engine mode, ships a screening-not-diagnosis disclaimer.
- **Robustness:** quality gating rejects unusable captures; every added stage
  degrades gracefully; demo never crashes.
- **Real ML:** standard EfficientNet + APTOS training pipeline included, not a
  black box, with calibration and OOD fitting built into the same script.

---

## Limitations & disclaimer

RetinaScan is a **screening aid, not a diagnostic device**. Predictions are
probabilistic and must be confirmed by a qualified clinician. Demo-mode (classical
CV) output is a rough indicator only. Accuracy depends heavily on capture quality
and the model/data used for training. Do not use as the sole basis for any
medical decision.

Specific to the newer features, stated plainly rather than glossed over:

- The **multi-condition screens are classical morphometry, not trained models.**
  VCDR, AVR and clarity are geometric proxies measured from a single 2-D colour
  photo; they have not been validated against clinician labels here. They are
  positioned as "worth a second look", not as diagnoses, and the UI says so.
- **Glaucoma cannot be diagnosed from cup-to-disc ratio alone** — it needs visual
  fields and IOP. A high VCDR is a referral trigger, nothing more.
- **AVR needs a disc-centred capture.** Off-centre photos give an unreliable
  arteriole/venule split, in which case the finding reports itself unavailable
  rather than guessing.
- **Cataract and defocus look the same to the camera.** When the capture is not
  sharp, low clarity is reported as "recapture first", not as opacity.
- The **feature-space OOD test needs `models/ood_stats.npz`** from a training run;
  without it only the classical fundus-likelihood check is active.
- Abstention thresholds in `config.py` are **reasoned defaults, not tuned on a
  validation set.** On real data you would pick them from an accuracy/coverage
  curve — how much accuracy you buy per percent of images deferred.
