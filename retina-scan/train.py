"""
Optional: train a real DR grader on the APTOS 2019 dataset (or any CSV of
image,label pairs) and produce `models/dr_model.pth` that app.py loads
automatically for genuine on-device CNN grading.

APTOS 2019 layout (Kaggle "aptos2019-blindness-detection"):
    train.csv         -> columns: id_code, diagnosis (0..4)
    train_images/     -> <id_code>.png

Example:
    python train.py --data-dir path/to/train_images --csv path/to/train.csv \
                    --epochs 8 --batch-size 16

Generic CSV:
    python train.py --data-dir imgs --csv labels.csv --img-col file \
                    --label-col grade --ext ""
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import timm

import config
import preprocessing
import ood as ood_mod
import uncertainty


class FundusDataset(Dataset):
    def __init__(self, rows, data_dir, ext, img_size, train=False):
        self.rows = rows
        self.data_dir = data_dir
        self.ext = ext
        self.img_size = img_size
        self.train = train
        self.mean = np.array(config.IMAGENET_MEAN, dtype="float32")
        self.std = np.array(config.IMAGENET_STD, dtype="float32")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        name, label = self.rows[idx]
        path = os.path.join(self.data_dir, f"{name}{self.ext}")
        img = preprocessing.preprocess(path, img_size=self.img_size, enhance=True)
        if self.train:
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :]
            if np.random.rand() < 0.5:
                img = img[::-1, :, :]
            k = int(np.random.randint(0, 4))
            if k:
                img = np.rot90(img, k)
        x = (np.ascontiguousarray(img).astype("float32") / 255.0 - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x), int(label)


def load_rows(csv_path, img_col, label_col):
    import csv
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        ic = img_col if img_col in cols else cols[0]
        lc = label_col if label_col in cols else cols[1]
        for r in reader:
            rows.append((r[ic], int(float(r[lc]))))
    return rows


def quadratic_kappa(y_true, y_pred, n=config.NUM_CLASSES):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    O = np.zeros((n, n))
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1
    w = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            w[i, j] = ((i - j) ** 2) / ((n - 1) ** 2)
    act = np.bincount(y_true, minlength=n)
    pred = np.bincount(y_pred, minlength=n)
    E = np.outer(act, pred) / max(len(y_true), 1)
    E = E / max(E.sum(), 1e-9) * O.sum()
    denom = (w * E).sum()
    return 1 - (w * O).sum() / denom if denom > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img-col", default="id_code")
    ap.add_argument("--label-col", default="diagnosis")
    ap.add_argument("--ext", default=".png")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--backbone", default=config.BACKBONE)
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--out", default=str(config.LOCAL_CHECKPOINT))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-calibrate", action="store_true",
                    help="Skip temperature scaling on the validation split.")
    ap.add_argument("--no-fit-ood", action="store_true",
                    help="Skip writing models/ood_stats.npz for the "
                         "feature-space out-of-distribution detector.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    rows = load_rows(args.csv, args.img_col, args.label_col)
    print(f"Loaded {len(rows)} labelled images")
    ds = FundusDataset(rows, args.data_dir, args.ext, args.img_size, train=True)
    n_val = max(1, int(len(ds) * args.val_split))
    n_tr = len(ds) - n_val
    g = torch.Generator().manual_seed(42)
    tr_ds, va_ds = random_split(ds, [n_tr, n_val], generator=g)
    # validation should not augment
    va_ds.dataset = FundusDataset(rows, args.data_dir, args.ext, args.img_size, train=False)

    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=(device == "cuda"))
    va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=(device == "cuda"))

    # class-weighted loss for imbalance
    labels = np.array([r[1] for r in rows])
    counts = np.bincount(labels, minlength=config.NUM_CLASSES).astype("float32")
    weights = (counts.sum() / (counts + 1e-6))
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)

    model = timm.create_model(args.backbone, pretrained=True,
                              num_classes=config.NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(weight=weights)

    best_kappa = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for i, (x, y) in enumerate(tr):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
            if i % 20 == 0:
                print(f"  epoch {epoch} step {i}/{len(tr)} loss {loss.item():.3f}")
        sched.step()

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for x, y in va:
                out = model(x.to(device))
                preds += out.argmax(1).cpu().tolist()
                gts += y.tolist()
        acc = float(np.mean(np.array(preds) == np.array(gts)))
        kappa = quadratic_kappa(gts, preds)
        print(f"[epoch {epoch}] train_loss {running/max(len(tr),1):.3f} "
              f"val_acc {acc:.3f} val_QWK {kappa:.3f}")

        if kappa >= best_kappa:
            best_kappa = kappa
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.out)
            print(f"  ✓ saved best -> {args.out} (QWK {kappa:.3f})")

    print(f"Done. Best val QWK: {best_kappa:.3f}. Checkpoint: {args.out}")

    # ----------------------------------------------------------------------
    # Post-training: calibration + OOD reference statistics.
    #
    # Both are computed on the held-out validation split so they describe the
    # model's behaviour on data it did not fit. Without these two artefacts the
    # app still runs, but it has to report confidences as uncalibrated and it
    # can only do the classical (non-feature-space) OOD check.
    # ----------------------------------------------------------------------
    model.load_state_dict(torch.load(args.out, map_location=device))
    model.eval().to(device)

    val_logits, val_labels, val_feats = [], [], []
    with torch.no_grad():
        for x, y in va:
            x = x.to(device)
            feats = model.forward_features(x)
            pooled = feats.mean(dim=(2, 3)) if feats.ndim == 4 else feats.mean(dim=1)
            val_logits.append(model.forward_head(feats).cpu().numpy())
            val_feats.append(pooled.cpu().numpy())
            val_labels.append(np.asarray(y))
    if val_logits:
        val_logits = np.concatenate(val_logits, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_feats = np.concatenate(val_feats, axis=0)
    else:
        val_logits = np.zeros((0, config.NUM_CLASSES))

    temperature = 1.0
    if not args.no_calibrate and len(val_logits):
        temperature = uncertainty.fit_temperature(val_logits, val_labels)
        print(f"Calibration: temperature T = {temperature:.4f} "
              f"({'softening overconfident logits' if temperature > 1 else 'sharpening'})")
        # Re-save as a dict so the app picks the temperature up automatically.
        torch.save({"state_dict": model.state_dict(),
                    "temperature": float(temperature),
                    "backbone": args.backbone,
                    "img_size": args.img_size,
                    "val_qwk": float(best_kappa)}, args.out)
        print(f"  ✓ checkpoint re-saved with calibration -> {args.out}")

    if not args.no_fit_ood and len(val_feats):
        try:
            stats = ood_mod.fit_stats(val_feats, out_path=str(config.OOD_STATS_PATH))
            print(f"  ✓ OOD reference stats -> {config.OOD_STATS_PATH} "
                  f"(dim {val_feats.shape[1]}, "
                  f"97.5th-pct distance {stats['ref_distance']:.1f})")
        except Exception as exc:
            print(f"  ! could not fit OOD stats: {exc}")


if __name__ == "__main__":
    main()
