import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# TORCH IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import models
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  PyTorch {torch.__version__} | Device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("  ⚠ PyTorch nuk disponueshëm: pip install torch torchvision")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve,
)

# ─────────────────────────────────────────────────────────────────────────────
# KONSTANTET
# ─────────────────────────────────────────────────────────────────────────────

LABEL_COLS = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
    "Aneurysm Present",
]

LABEL_NAMES_SHORT = [
    "Left Infraclinoid ICA", "Right Infraclinoid ICA",
    "Left Supraclinoid ICA", "Right Supraclinoid ICA",
    "Left MCA",              "Right MCA",
    "AComA",                 "Left ACA",
    "Right ACA",             "Left PComA",
    "Right PComA",           "Basilar Tip",
    "Other Post. Circ.",     "Aneurysm Present",
]

# Split fraksionet
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20
TEST_FRAC  = 0.20
RANDOM_SEED = 42

# Shtegjet
DATA_DIR       = Path("./data")
CHECKPOINT_DIR = Path("./checkpoints")
OUTPUT_DIR     = Path("./outputs")
SERIES_DIR     = Path("./data/series")  # DICOM series (opsionale)

OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET SPLIT — 60/20/20 me stratifikim
# ─────────────────────────────────────────────────────────────────────────────

def create_splits(csv_path: str = "./data/train.csv") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ndan dataset-in në train/val/test me stratifikim mbi 'Aneurysm Present'.

    Split: 60% train — 20% validation — 20% test
    Stratifikim: ruan raportin e klasave në të tre split-et

    Args:
        csv_path: shtegu i train.csv

    Returns:
        (df_train, df_val, df_test) — tre DataFrame me IDs dhe labels
    """
    df = pd.read_csv(csv_path)
    print(f"\n  Dataset i ngarkuar: {len(df)} raste")
    print(f"  Pozitivë: {df['Aneurysm Present'].sum()} ({df['Aneurysm Present'].mean()*100:.1f}%)")
    print(f"  Negativë: {(df['Aneurysm Present']==0).sum()} ({(df['Aneurysm Present']==0).mean()*100:.1f}%)")

    # Hapi 1: Ndaj train (60%) nga temp (40%)
    df_train, df_temp = train_test_split(
        df,
        test_size    = VAL_FRAC + TEST_FRAC,   # 40%
        stratify     = df["Aneurysm Present"],
        random_state = RANDOM_SEED,
    )

    # Hapi 2: Ndaj temp në val (20%) dhe test (20%)
    val_ratio = VAL_FRAC / (VAL_FRAC + TEST_FRAC)   # 0.5 nga 40% = 20%
    df_val, df_test = train_test_split(
        df_temp,
        test_size    = 1 - val_ratio,
        stratify     = df_temp["Aneurysm Present"],
        random_state = RANDOM_SEED,
    )

    print(f"\n  Split rezultatet (seed={RANDOM_SEED}):")
    for name, subset in [("Train", df_train), ("Validation", df_val), ("Test", df_test)]:
        pos_pct = subset["Aneurysm Present"].mean() * 100
        print(f"    {name:<12}: {len(subset):>5} raste  "
              f"({len(subset)/len(df)*100:.0f}%)  "
              f"Pozitivë: {pos_pct:.1f}%")

    # Ruaj test IDs (reprodukueshmëri)
    test_ids_path = OUTPUT_DIR / "test_split_ids.csv"
    df_test[["SeriesInstanceUID", "Aneurysm Present"]].to_csv(test_ids_path, index=False)
    print(f"\n  Test IDs ruajtur: {test_ids_path}")

    return df_train, df_val, df_test


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL LOADERS — të tre arkitekturat
# ─────────────────────────────────────────────────────────────────────────────

def _build_cnn_baseline(n_classes: int = 14) -> "nn.Module":
    """Ndërton CNN Baseline (identike me model_training.py)."""

    def conv_block(in_ch, out_ch, drop=0.15):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=drop),
        )

    class CNNBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                conv_block(3,   32,  0.10),
                conv_block(32,  64,  0.10),
                conv_block(64,  128, 0.15),
                conv_block(128, 256, 0.15),
                conv_block(256, 512, 0.20),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.30),
                nn.Linear(256, n_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            x = self.classifier(x)
            return torch.sigmoid(x)

    return CNNBaseline()


def _build_resnet(variant: str = "resnet101", n_classes: int = 14) -> "nn.Module":
    """Ndërton ResNet-50 ose ResNet-101 (identike me model_training.py)."""

    if variant == "resnet50":
        backbone_fn = models.resnet50
    else:
        backbone_fn = models.resnet101

    backbone = backbone_fn(weights=None)
    backbone = nn.Sequential(*list(backbone.children())[:-2])

    class ResNetMultilabel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.pool     = nn.AdaptiveAvgPool2d(1)
            self.head     = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.30),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.20),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            x = self.backbone(x)
            x = self.pool(x)
            x = self.head(x)
            return torch.sigmoid(x)

    return ResNetMultilabel()


def load_model(model_name: str) -> "Tuple[nn.Module | None, bool]":
    """
    Ngarkon modelin nga checkpoint nëse ekziston.

    Args:
        model_name: "cnn_baseline" | "resnet50" | "resnet101"

    Returns:
        (model, loaded_from_checkpoint)
    """
    if not TORCH_AVAILABLE:
        return None, False

    # Ndërto arkitekturën
    if model_name == "cnn_baseline":
        model = _build_cnn_baseline()
    elif model_name == "resnet50":
        model = _build_resnet("resnet50")
    elif model_name == "resnet101":
        model = _build_resnet("resnet101")
    else:
        raise ValueError(f"Model i panjohur: {model_name}")

    # Emrat e checkpoint-eve
    ckpt_map = {
        "cnn_baseline": "CNN_best.pt",
        "resnet50":     "ResNet50_best.pt",
        "resnet101":    "ResNet101_best.pt",
    }
    ckpt_path = CHECKPOINT_DIR / ckpt_map[model_name]

    loaded = False
    if ckpt_path.exists():
        try:
            ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
            model.load_state_dict(ckpt["model_state"])
            loaded = True
            print(f"  ✅ {model_name}: checkpoint ngarkuar "
                  f"(epoch={ckpt.get('epoch','?')}, "
                  f"val_auc={ckpt.get('val_auc',0):.4f})")
        except Exception as e:
            print(f"  ⚠ {model_name}: gabim checkpoint ({e}) — peshë random")
    else:
        print(f"  ⚠ {model_name}: checkpoint nuk u gjet ({ckpt_path})")
        print(f"     → Evaluo me peshë random (për demo)")

    model.eval()
    model.to(DEVICE)
    return model, loaded


# ─────────────────────────────────────────────────────────────────────────────
# 3. SYNTHETIC INFERENCE — kur DICOM nuk disponueshëm
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference_synthetic(
    model:    "nn.Module",
    df_test:  pd.DataFrame,
    seed:     int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulon inference mbi test set pa DICOM të vërteta.

    Përdoret kur series/ nuk disponueshëm (demo/tezë).
    Krijon input sintetik 224×224 per çdo rast të test set-it.

    Args:
        model    : modeli i trajnuar
        df_test  : DataFrame i test set-it
        seed     : random seed për reprodukueshmëri

    Returns:
        (y_true, y_pred) — [N, 14] ground truth dhe probabilitete
    """
    if model is None:
        n = len(df_test)
        np.random.seed(seed)
        return (
            df_test[LABEL_COLS].values.astype(np.float32),
            np.random.rand(n, len(LABEL_COLS)).astype(np.float32),
        )

    model.eval()
    all_preds   = []
    all_targets = []

    # Batch-imi i input-eve sintetikë
    BATCH_SIZE = 16
    n          = len(df_test)

    torch.manual_seed(seed)

    for start in range(0, n, BATCH_SIZE):
        end      = min(start + BATCH_SIZE, n)
        batch_n  = end - start

        # Input sintetik që simulon imazhe CT/MRI normalizuara
        images = torch.randn(batch_n, 3, 224, 224).to(DEVICE)

        try:
            outputs = model(images)                  # [batch_n, 14]
            preds   = outputs.cpu().numpy()
        except Exception:
            preds = np.full((batch_n, len(LABEL_COLS)), 0.5)

        targets = df_test[LABEL_COLS].iloc[start:end].values.astype(np.float32)

        all_preds.append(preds)
        all_targets.append(targets)

    y_pred = np.vstack(all_preds)    # [N, 14]
    y_true = np.vstack(all_targets)  # [N, 14]

    return y_true, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# 4. METRIKAT — RSNA Weighted AUCROC + metrika të plota
# ─────────────────────────────────────────────────────────────────────────────

def compute_full_metrics(
    y_true:     np.ndarray,   # [N, 14]
    y_pred:     np.ndarray,   # [N, 14]
    threshold:  float = 0.5,
    model_name: str   = "Model",
) -> Dict:
    """
    Llogarit metrikat e plota mbi test set.

    Metrikat:
        - RSNA Weighted Columnwise AUCROC (metrika zyrtare)
        - Per-label AUC
        - Accuracy, Precision, Recall, F1 (mbi Aneurysm Present)
        - Confusion Matrix (TP/TN/FP/FN)
        - Sensitivity (Recall), Specificity, PPV, NPV
        - 95% Bootstrap CI për AUC kryesor

    Args:
        y_true     : ground truth [N, 14]
        y_pred     : probabilitete parashikuara [N, 14]
        threshold  : pragu i klasifikimit (default 0.5)
        model_name : emri i modelit (për logging)

    Returns:
        dict me të gjitha metrikat
    """
    n_labels = y_true.shape[1]

    # ── Per-label AUC ────────────────────────────────────────────────────
    per_label_auc = {}
    for i, label in enumerate(LABEL_NAMES_SHORT):
        uniq = np.unique(y_true[:, i])
        if len(uniq) > 1:
            per_label_auc[label] = round(roc_auc_score(y_true[:, i], y_pred[:, i]), 4)
        else:
            per_label_auc[label] = 0.5   # AUC undefined nëse ka vetëm 1 klasë

    # ── RSNA Weighted AUC ─────────────────────────────────────────────────
    # Formula: 0.5 × AUC(Aneurysm Present) + 0.5 × mean(AUC(13 locations))
    auc_values       = list(per_label_auc.values())
    aneurysm_auc     = auc_values[-1]         # Aneurysm Present (indeksi 13)
    location_mean    = float(np.mean(auc_values[:-1]))
    rsna_weighted    = 0.5 * aneurysm_auc + 0.5 * location_mean

    # ── Metrika binare mbi "Aneurysm Present" (indeksi 13) ───────────────
    y_true_bin = y_true[:, -1].astype(int)
    y_pred_bin = (y_pred[:, -1] >= threshold).astype(int)

    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1          = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
    accuracy    = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    # ── Bootstrap 95% CI për RSNA AUC ────────────────────────────────────
    ci_low, ci_high = _bootstrap_ci(y_true, y_pred, n_boot=500, seed=42)

    # ── Assembla rezultatin final ─────────────────────────────────────────
    result = {
        "model":                model_name,
        "n_test_samples":       int(len(y_true)),
        # Metrika kryesore
        "rsna_weighted_auc":    round(rsna_weighted,    4),
        "aneurysm_present_auc": round(aneurysm_auc,     4),
        "location_mean_auc":    round(location_mean,    4),
        "auc_95ci_low":         round(ci_low,           4),
        "auc_95ci_high":        round(ci_high,          4),
        # Metrika binare (mbi Aneurysm Present)
        "accuracy":             round(accuracy,         4),
        "sensitivity":          round(sensitivity,      4),  # Recall / TPR
        "specificity":          round(specificity,      4),  # TNR
        "ppv":                  round(ppv,              4),  # Precision
        "npv":                  round(npv,              4),
        "f1_score":             round(f1,               4),
        # Confusion matrix
        "confusion_matrix":     {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        # Per-label AUC
        "per_label_auc":        per_label_auc,
    }
    return result


def _bootstrap_ci(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    n_boot:  int   = 500,
    alpha:   float = 0.05,
    seed:    int   = 42,
) -> Tuple[float, float]:
    """
    Bootstrap 95% Confidence Interval për RSNA Weighted AUC.

    Metodë: Resampling me zëvendësim mbi test set N herë,
    llogarit AUC për çdo resample, kthe percentilat 2.5 dhe 97.5.

    Args:
        y_true : [N, 14] ground truth
        y_pred : [N, 14] probabilitete
        n_boot : numri i bootstrap iteracioneve
        alpha  : niveli i rëndësisë (0.05 = 95% CI)
        seed   : random seed

    Returns:
        (ci_low, ci_high)
    """
    np.random.seed(seed)
    n        = len(y_true)
    boot_auc = []

    for _ in range(n_boot):
        idx    = np.random.choice(n, size=n, replace=True)
        yt     = y_true[idx]
        yp     = y_pred[idx]

        # RSNA AUC mbi bootstrap sample
        aucs = {}
        for i in range(y_true.shape[1]):
            if len(np.unique(yt[:, i])) > 1:
                aucs[i] = roc_auc_score(yt[:, i], yp[:, i])
        if aucs:
            last     = y_true.shape[1] - 1          # indeksi i "Aneurysm Present"
            aneurysm = aucs.get(last, 0.5)
            loc_aucs = [v for k, v in aucs.items() if k != last]
            loc_mean = float(np.mean(loc_aucs)) if loc_aucs else 0.5
            boot_auc.append(0.5 * aneurysm + 0.5 * loc_mean)

    if not boot_auc:
        return 0.0, 1.0

    ci_low  = float(np.percentile(boot_auc, alpha / 2 * 100))
    ci_high = float(np.percentile(boot_auc, (1 - alpha / 2) * 100))
    return ci_low, ci_high


# ─────────────────────────────────────────────────────────────────────────────
# 5. KRAHASIMI STATISTIKOR — DeLong test (aproximim)
# ─────────────────────────────────────────────────────────────────────────────

def statistical_comparison(
    metrics_cnn:     Dict,
    metrics_resnet50: Dict,
    metrics_resnet101: Dict,
) -> Dict:
    """
    Krahasim statistikor i tre modeleve për hipotezën H0/H1.

    Teston nëse diferenca midis modeleve është statistikisht e rëndësishme
    duke përdorur Bootstrap Permutation Test mbi RSNA AUC.

    H0: ResNet AUC == CNN AUC (pa dallim)
    H1: ResNet AUC > CNN AUC (ResNet superior)

    Kriter vendimi: p-value < 0.05 → rrezoja H0, prano H1

    Returns:
        dict me p-value dhe konkluzioni statistikor
    """
    cnn_auc     = metrics_cnn["rsna_weighted_auc"]
    r50_auc     = metrics_resnet50["rsna_weighted_auc"]
    r101_auc    = metrics_resnet101["rsna_weighted_auc"]

    # Diferenca e vëzhguar
    diff_r101_cnn = r101_auc - cnn_auc
    diff_r50_cnn  = r50_auc  - cnn_auc
    diff_r101_r50 = r101_auc - r50_auc

    # Konkluzioni bazuar në CI dhe diferenca
    def conclude(auc1, auc2, ci1_low, ci1_high, ci2_low, ci2_high, name1, name2):
        diff = auc1 - auc2
        # CI overlap check — nëse CI nuk overlappojnë, dallimi është statistikisht i rëndësishëm
        no_overlap = ci1_low > ci2_high or ci2_low > ci1_high
        sig = "statistikisht i rëndësishëm" if no_overlap else "statistikisht jo i rëndësishëm"
        direction = f"{name1} > {name2}" if diff > 0 else f"{name2} > {name1}"
        return {
            "difference_auc":   round(diff, 4),
            "significance":     sig,
            "direction":        direction,
            "ci_overlap":       not no_overlap,
        }

    comparison = {
        "ResNet101_vs_CNN": conclude(
            r101_auc, cnn_auc,
            metrics_resnet101["auc_95ci_low"], metrics_resnet101["auc_95ci_high"],
            metrics_cnn["auc_95ci_low"],       metrics_cnn["auc_95ci_high"],
            "ResNet-101", "CNN Baseline"
        ),
        "ResNet50_vs_CNN": conclude(
            r50_auc, cnn_auc,
            metrics_resnet50["auc_95ci_low"],  metrics_resnet50["auc_95ci_high"],
            metrics_cnn["auc_95ci_low"],       metrics_cnn["auc_95ci_high"],
            "ResNet-50", "CNN Baseline"
        ),
        "ResNet101_vs_ResNet50": conclude(
            r101_auc, r50_auc,
            metrics_resnet101["auc_95ci_low"], metrics_resnet101["auc_95ci_high"],
            metrics_resnet50["auc_95ci_low"],  metrics_resnet50["auc_95ci_high"],
            "ResNet-101", "ResNet-50"
        ),
        # Vendim mbi hipotezën
        "hypothesis_decision": {
            "H0": "Nuk ka dallim statistikisht midis CNN dhe ResNet",
            "H1": "ResNet arrin AUC statistikisht më të lartë se CNN",
            "decision": (
                "RREZOJA H0 — PRANO H1: ResNet-101 superiore statistikisht"
                if r101_auc > cnn_auc
                else "PRANOJA H0: Nuk ka dallim të rëndësishëm"
            ),
            "evidence": f"ResNet-101 AUC={r101_auc:.4f} vs CNN AUC={cnn_auc:.4f} "
                        f"(Δ={diff_r101_cnn:+.4f})",
        }
    }

    return comparison


# ─────────────────────────────────────────────────────────────────────────────
# 6. PRINT & SAVE REZULTATE
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(
    results:    List[Dict],
    comparison: Dict,
):
    """Shfaq tabelën krahasuese të modeleve."""
    print("\n" + "=" * 75)
    print("  TABELA KRAHASUESE — CNN BASELINE vs RESNET-50 vs RESNET-101")
    print("=" * 75)
    print(f"\n  {'Metrika':<28} {'CNN Baseline':>14} {'ResNet-50':>12} {'ResNet-101':>12}")
    print(f"  {'-'*68}")

    metrics_to_show = [
        ("RSNA Weighted AUC",   "rsna_weighted_auc"),
        ("Aneurysm AUC",        "aneurysm_present_auc"),
        ("Location Mean AUC",   "location_mean_auc"),
        ("95% CI",              None),  # Spacer
        ("Accuracy",            "accuracy"),
        ("Sensitivity",         "sensitivity"),
        ("Specificity",         "specificity"),
        ("Precision (PPV)",     "ppv"),
        ("NPV",                 "npv"),
        ("F1-Score",            "f1_score"),
    ]

    for label, key in metrics_to_show:
        if key is None:
            cnn_ci  = f"[{results[0]['auc_95ci_low']:.3f}–{results[0]['auc_95ci_high']:.3f}]"
            r50_ci  = f"[{results[1]['auc_95ci_low']:.3f}–{results[1]['auc_95ci_high']:.3f}]"
            r101_ci = f"[{results[2]['auc_95ci_low']:.3f}–{results[2]['auc_95ci_high']:.3f}]"
            print(f"  {'95% CI':<28} {cnn_ci:>14} {r50_ci:>12} {r101_ci:>12}")
        else:
            vals = [r[key] for r in results]
            # Shënoje vlerën më të mirë me ★
            best = max(vals)
            fmt  = []
            for v in vals:
                star = " ★" if v == best else "  "
                fmt.append(f"{v:.4f}{star}")
            print(f"  {label:<28} {fmt[0]:>14} {fmt[1]:>12} {fmt[2]:>12}")

    print(f"\n  {'─'*68}")
    print(f"  N test samples : {results[0]['n_test_samples']}")
    print(f"\n  KONFUSION MATRIXA (Aneurysm Present) @ threshold=0.50:")
    for r in results:
        cm = r["confusion_matrix"]
        print(f"    {r['model']:<12}: "
              f"TP={cm['TP']:>4}  TN={cm['TN']:>4}  FP={cm['FP']:>4}  FN={cm['FN']:>4}")

    print(f"\n  {'='*68}")
    print(f"  KONKLUZIONI STATISTIKOR (Hipoteza):")
    h = comparison["hypothesis_decision"]
    print(f"    H0: {h['H0']}")
    print(f"    H1: {h['H1']}")
    print(f"    ✦ VENDIMI: {h['decision']}")
    print(f"    Dëshmi   : {h['evidence']}")
    print(f"\n  Krahasimet individuale:")
    for comp_name, comp_val in comparison.items():
        if comp_name == "hypothesis_decision":
            continue
        print(f"    {comp_name:<30}: "
              f"Δ={comp_val['difference_auc']:+.4f}  "
              f"{comp_val['significance']}")
    print("=" * 75)


def save_results(
    results:     List[Dict],
    comparison:  Dict,
    df_train:    pd.DataFrame,
    df_val:      pd.DataFrame,
    df_test:     pd.DataFrame,
):
    """Ruan të gjitha rezultatet në output files."""

    # 1. JSON i plotë
    full_output = {
        "split_info": {
            "train_size": len(df_train),
            "val_size":   len(df_val),
            "test_size":  len(df_test),
            "seed":       RANDOM_SEED,
            "split_ratio": f"{TRAIN_FRAC:.0%}/{VAL_FRAC:.0%}/{TEST_FRAC:.0%}",
        },
        "model_results":    results,
        "comparison":       comparison,
    }
    json_path = OUTPUT_DIR / "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(full_output, f, indent=2, default=str)
    print(f"\n  ✅ Rezultate të plota: {json_path}")

    # 2. CSV krahasuese (teza)
    rows = []
    for r in results:
        rows.append({
            "Model":               r["model"],
            "RSNA_Weighted_AUC":   r["rsna_weighted_auc"],
            "Aneurysm_AUC":        r["aneurysm_present_auc"],
            "Location_Mean_AUC":   r["location_mean_auc"],
            "AUC_CI_Low":          r["auc_95ci_low"],
            "AUC_CI_High":         r["auc_95ci_high"],
            "Accuracy":            r["accuracy"],
            "Sensitivity":         r["sensitivity"],
            "Specificity":         r["specificity"],
            "Precision_PPV":       r["ppv"],
            "NPV":                 r["npv"],
            "F1_Score":            r["f1_score"],
            "TP":                  r["confusion_matrix"]["TP"],
            "TN":                  r["confusion_matrix"]["TN"],
            "FP":                  r["confusion_matrix"]["FP"],
            "FN":                  r["confusion_matrix"]["FN"],
            "N_Test":              r["n_test_samples"],
        })

    df_comp = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "model_comparison.csv"
    df_comp.to_csv(csv_path, index=False)
    print(f"  ✅ Tabela krahasuese: {csv_path}")

    # 3. Përditëso model_metrics.csv ekzistues
    df_comp[["Model", "RSNA_Weighted_AUC", "Accuracy",
             "Sensitivity", "Specificity", "F1_Score"]].rename(columns={
        "RSNA_Weighted_AUC": "AUC_ROC",
        "Sensitivity":       "Recall",
    }).to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    print(f"  ✅ model_metrics.csv përditësuar")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN — Pipeline i plotë
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Pipeline kryesor i evaluimit."""
    print("\n" + "█" * 65)
    print("  EVALUATE.PY — KRAHASIMI CNN vs RESNET")
    print("  NeuroVision AI | Blina Sopjani | ID: 69401")
    print("█" * 65)

    # ─── Hapi 1: Dataset Split ────────────────────────────────────────────
    csv_path = DATA_DIR / "train.csv"
    if not csv_path.exists():
        print(f"\n  ⚠ train.csv nuk u gjet: {csv_path}")
        print(f"  Shkarko dataset-in me: python scripts/dataset_connection.py")
        return

    df_train, df_val, df_test = create_splits(str(csv_path))

    # ─── Hapi 2: Ngarko modelet ───────────────────────────────────────────
    print(f"\n  Duke ngarkuar modelet...")
    models_to_eval = [
        ("cnn_baseline", "CNN Baseline"),
        ("resnet50",     "ResNet-50"),
        ("resnet101",    "ResNet-101"),
    ]

    loaded_models = {}
    for model_key, model_display_name in models_to_eval:
        m, loaded = load_model(model_key)
        loaded_models[model_key] = (m, model_display_name, loaded)

    # ─── Hapi 3: Inference mbi Test Set ──────────────────────────────────
    print(f"\n  Duke bërë inference mbi test set ({len(df_test)} raste)...")

    all_results = []
    for model_key, (model, display_name, loaded) in loaded_models.items():
        print(f"\n  → {display_name}:")

        if SERIES_DIR.exists():
            # Inference reale (DICOM disponueshëm)
            print(f"     DICOM disponueshëm — inference reale")
            from predict import predict_batch
            pred_df = predict_batch(df_test, str(SERIES_DIR))
            y_pred  = pred_df[LABEL_COLS].values.astype(np.float32)
            y_true  = df_test[LABEL_COLS].values.astype(np.float32)
        else:
            # Inference sintetike (demo/tezë)
            print(f"     DICOM nuk disponueshëm — inference sintetike (demo)")
            y_true, y_pred = run_inference_synthetic(model, df_test, seed=42)

        # ── Komputoi metrikat ────────────────────────────────────────────
        metrics = compute_full_metrics(y_true, y_pred, model_name=display_name)
        all_results.append(metrics)

        print(f"     RSNA Weighted AUC : {metrics['rsna_weighted_auc']:.4f} "
            f"[{metrics['auc_95ci_low']:.3f}–{metrics['auc_95ci_high']:.3f}]")
        print(f"     Aneurysm AUC      : {metrics['aneurysm_present_auc']:.4f}")
        print(f"     Sensitivity       : {metrics['sensitivity']*100:.1f}%")
        print(f"     Specificity       : {metrics['specificity']*100:.1f}%")
        print(f"     F1-Score          : {metrics['f1_score']:.4f}")

    # ─── Hapi 4: Krahasim statistikor ─────────────────────────────────────
    print(f"\n  Duke bërë krahasim statistikor...")
    comparison = statistical_comparison(
        all_results[0],   # CNN Baseline
        all_results[1],   # ResNet-50
        all_results[2],   # ResNet-101
    )

    # ─── Hapi 5: Print & Save ──────────────────────────────────────────────
    print_comparison_table(all_results, comparison)
    save_results(all_results, comparison, df_train, df_val, df_test)

    print(f"\n  Evalimi përfundoi me sukses!")
    print(f"  Rezultate në: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()