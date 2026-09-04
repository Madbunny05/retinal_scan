# Put your trained weights here

Drop a checkpoint named **`dr_model.pth`** in this folder and `app.py` will
load it automatically for real CNN grading (no internet needed).

Accepted formats:
- a raw `state_dict` saved with `torch.save(model.state_dict(), ...)`, or
- a dict containing a `"state_dict"` key.

The weights must match the backbone set in `../config.py` (`BACKBONE`,
default `efficientnet_b0`) and have **5 output classes** (ICDR grades 0-4).

Easiest way to produce one: `python ../train.py --help` (APTOS 2019).

If no checkpoint is found here, the app runs in **demo mode** using the
classical computer-vision lesion grader in `lesion_analysis.py` — the UI stays
fully functional so your demo never breaks.
