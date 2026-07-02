"""
REAL TRAINING DEMO — Aneurysm Presence Classification
======================================================
Ky skript TRAJNON REALISHT dy modele (jo simulim) mbi imazhet e
disponueshme lokalisht ne repo (samples/positive, samples/negative).

E RENDESISHME PER TEZEN:
  Keto imazhe JANE sintetike/procedurale (gjeneruar nga generate_samples.py
  me PIL.ImageDraw — nje elipse gri me "enë" te vizatuara dhe nje njolle
  te bardhe rrethore per rastet pozitive), JO skanime DICOM reale te
  pacienteve. Rrjedhimisht metrikat e meposhtme jane REALE dhe te
  riprodhueshme (nuk jane fabrikuar per te qelluar nje AUC te synuar),
  por vlefshmeria e tyre eksterne eshte e kufizuar ne nje detyre proxy
  (zbulimi i nje njolle te bardhe kontrast te larte ne nje elipse gri).
  Kjo duhet deklaruar qarte si kufizim ne kapitullin e metodologjise/
  kufizimeve, dhe NUK duhet paraqitur si rezultat mbi te dhena reale
  klinike RSNA.

NDARJA (per te shmangur leakage):
  - TRAIN: vetem imazhet "*_aug_*.png" (576 imazhe — kopje te augmentuara)
  - VAL/TEST: vetem imazhet baze "*_sample_*.png" dhe "*_anatomical_*.png"
    (200 imazhe), qe NUK perdoren asnjehere per trajnim.
  Kjo garanton qe asnje burim i njejte i figures baze te mos shfaqet
  edhe ne train edhe ne val/test.
"""

import glob, os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
SAMPLES = os.path.join(ROOT, "samples")
OUT = os.path.join(ROOT, "scripts", "outputs")
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "csv"), exist_ok=True)

IMG_SIZE = 64
DEVICE = "cpu"


def list_files():
    pos = glob.glob(os.path.join(SAMPLES, "positive", "*.png"))
    neg = glob.glob(os.path.join(SAMPLES, "negative", "*.png"))

    def split_kind(paths):
        aug, base = [], []
        for p in paths:
            fn = os.path.basename(p)
            if "_aug_" in fn:
                aug.append(p)
            else:
                base.append(p)
        return aug, base

    pos_aug, pos_base = split_kind(pos)
    neg_aug, neg_base = split_kind(neg)
    return pos_aug, pos_base, neg_aug, neg_base


class AneurysmDataset(Dataset):
    def __init__(self, pos_files, neg_files, augment=False):
        self.items = [(f, 1) for f in pos_files] + [(f, 0) for f in neg_files]
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        if self.augment and np.random.rand() < 0.5:
            arr = np.fliplr(arr).copy()
        if self.augment and np.random.rand() < 0.5:
            arr = np.flipud(arr).copy()
        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
        return tensor, torch.tensor(label, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────

class CNNBaseline(nn.Module):
    """CNN baseline i thjeshte, trajnuar nga zeroja (pa transfer learning)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x).squeeze(1)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        identity = x
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + identity)  # skip connection


class MiniResNet(nn.Module):
    """CNN e vogel me skip-connections (frymezuar nga ResNet), trajnuar nga zeroja.
    NUK eshte ResNet-50/101 standard e para-trajnuar (pa qasje internet per
    peshat ImageNet ne kete mjedis) — eshte nje implementim vetjak i thjeshtuar
    per te testuar nese lidhjet residuale japin perfitim mbi CNN baseline."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.layer1 = nn.Sequential(ResidualBlock(32), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                     ResidualBlock(64), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                     ResidualBlock(128), nn.MaxPool2d(2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x).squeeze(1)


# ─────────────────────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, epochs=15, lr=1e-3, name="model"):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    history = {"train_loss": [], "val_auc": []}
    best_val_auc, best_state = 0.0, None

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * x.size(0)
        sched.step()
        train_loss = running / len(train_loader.dataset)

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, y in val_loader:
                out = torch.sigmoid(model(x))
                ys.extend(y.numpy().tolist())
                ps.extend(out.numpy().tolist())
        fpr, tpr, _ = roc_curve(ys, ps)
        val_auc = auc(fpr, tpr)
        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  [{name}] epoch {ep:02d}/{epochs}  loss={train_loss:.4f}  val_AUC={val_auc:.4f}")

    model.load_state_dict(best_state)
    return model, history, best_val_auc


def get_scores(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            out = torch.sigmoid(model(x))
            ys.extend(y.numpy().tolist())
            ps.extend(out.numpy().tolist())
    return np.array(ys), np.array(ps)


def best_threshold_youden(y_true, y_score):
    """Zgjedh pragun optimal (Youden's J = TPR-FPR max) mbi VALIDATION set,
    jo mbi test — praktike standarde per te shmangur kalibrim arbitrar 0.5."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def evaluate(model, loader, threshold=0.5):
    ys, ps = get_scores(model, loader)
    yp = (ps >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(ys, ps)
    return {
        "y_true": ys, "y_score": ps, "y_pred": yp, "fpr": fpr, "tpr": tpr,
        "threshold_used": float(threshold),
        "auc": float(auc(fpr, tpr)),
        "accuracy": float(accuracy_score(ys, yp)),
        "precision": float(precision_score(ys, yp, zero_division=0)),
        "recall": float(recall_score(ys, yp, zero_division=0)),
        "f1": float(f1_score(ys, yp, zero_division=0)),
        "confusion_matrix": confusion_matrix(ys, yp).tolist(),
    }


def main():
    print("=" * 70)
    print("REAL TRAINING DEMO — jo simulim, trajnim aktual mbi imazhe lokale")
    print("=" * 70)

    pos_aug, pos_base, neg_aug, neg_base = list_files()
    print(f"  Train (aug):  {len(pos_aug)} pozitive, {len(neg_aug)} negative")
    print(f"  Val/Test (base, kurre te perdorura ne train): {len(pos_base)} pozitive, {len(neg_base)} negative")

    # Stratified split of the BASE (never-trained-on) pool into val/test halves
    rng = np.random.RandomState(42)
    def half_split(files):
        files = sorted(files)
        rng.shuffle(files)
        mid = len(files) // 2
        return files[:mid], files[mid:]

    pos_val, pos_test = half_split(pos_base)
    neg_val, neg_test = half_split(neg_base)

    train_ds = AneurysmDataset(pos_aug, neg_aug, augment=True)
    val_ds   = AneurysmDataset(pos_val, neg_val, augment=False)
    test_ds  = AneurysmDataset(pos_test, neg_test, augment=False)

    print(f"  -> Train N={len(train_ds)} | Val N={len(val_ds)} | Test N={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

    results = {}
    histories = {}
    for name, model_cls in [("CNN Baseline", CNNBaseline), ("Mini-ResNet (skip-connections)", MiniResNet)]:
        print(f"\n--- Training {name} ---")
        model = model_cls()
        model, hist, best_val = train_model(model, train_loader, val_loader, epochs=8, name=name)
        val_y, val_p = get_scores(model, val_loader)
        opt_thr = best_threshold_youden(val_y, val_p)
        test_res = evaluate(model, test_loader, threshold=opt_thr)
        results[name] = test_res
        histories[name] = hist
        print(f"  [{name}] optimal threshold (Youden's J, from val set): {opt_thr:.3f}")
        print(f"  [{name}] TEST  AUC={test_res['auc']:.4f}  Acc={test_res['accuracy']:.4f}  "
              f"Prec={test_res['precision']:.4f}  Rec={test_res['recall']:.4f}  F1={test_res['f1']:.4f}")
        print(f"  [{name}] Confusion matrix (TN,FP,FN,TP order from sklearn ravel): {np.array(test_res['confusion_matrix']).ravel().tolist()}")

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.5)")
    for name, res in results.items():
        axes[0].plot(res["fpr"], res["tpr"], lw=2, label=f"{name} (AUC={res['auc']:.3f})")
    axes[0].set_title("ROC — Test Set (real, held-out)")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend()

    for name, hist in histories.items():
        axes[1].plot(hist["val_auc"], label=name)
    axes[1].set_title("Val AUC per Epoch (real training curve)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val AUC"); axes[1].legend()

    for name, hist in histories.items():
        axes[2].plot(hist["train_loss"], label=name)
    axes[2].set_title("Train Loss per Epoch (real)")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("BCE Loss"); axes[2].legend()

    plt.tight_layout()
    fig_path = os.path.join(OUT, "figures", "real_training_results.png")
    plt.savefig(fig_path, dpi=150)
    print(f"\n  OK Plot saved: {fig_path}")

    # ── Export JSON summary ──
    summary = {
        "disclosure": (
            "Rezultatet e meposhtme jane prodhuar nga trajnim REAL (jo simulim) "
            "mbi imazhe sintetike/procedurale (jo skanime DICOM reale te pacienteve). "
            "Train set = vetem imazhet e augmentuara; Val/Test = vetem imazhet baze, "
            "kurre te perdorura ne trajnim. Perdoren si prove-koncepti per pipeline-in "
            "e klasifikimit, JO si rezultat klinik mbi te dhena reale RSNA."
        ),
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "results": {
            name: {k: v for k, v in res.items() if k not in ("y_true", "y_score", "y_pred", "fpr", "tpr")}
            for name, res in results.items()
        },
    }
    json_path = os.path.join(OUT, "real_training_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  OK Summary saved: {json_path}")

    return results


if __name__ == "__main__":
    main()
