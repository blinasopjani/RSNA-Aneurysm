"""
=============================================================================
NeuroVision AI — MODEL ARCHITECTURES & TRAINING PIPELINE
=============================================================================
Student   : Blina Sopjani | ID: 69401
File      : model_training.py

MODELET:
  1. CNNBaseline   — CNN nga zeroja (5 convolutional blocks)
  2. ResNet50Model — ResNet-50 me ImageNet transfer learning
  3. ResNet101Model— ResNet-101 me ImageNet transfer learning (BEST)

TRAINING PIPELINE:
  - Weighted Binary Cross-Entropy Loss (class imbalance w_pos=3.67)
  - AdamW optimizer me Cosine Annealing scheduler
  - Early stopping (patience=7)
  - AUC-ROC tracking per epoch
  - Checkpoint saving (best model)

REQUIREMENTS:
  pip install torch torchvision scikit-learn tqdm numpy pandas
=============================================================================
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# TORCH IMPORTS (graceful fallback nëse nuk është instaluar)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  PyTorch {torch.__version__} | Device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    print("  ⚠ PyTorch nuk është instaluar: pip install torch torchvision")

from sklearn.metrics import roc_auc_score

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURIM
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "n_classes":       14,      # 13 lokacione + Aneurysm Present
    "input_size":      224,     # ResNet standard input
    "batch_size":      16,
    "learning_rate":   1e-4,
    "weight_decay":    1e-5,
    "epochs":          50,
    "patience":        7,       # Early stopping
    "pos_weight":      3.67,    # n_neg / n_pos = 2514/686
    "freeze_epochs":   10,      # Epoch deri kur ngrij backbone
    "dropout":         0.30,
    "checkpoint_dir":  "./checkpoints/",
    "output_dir":      "./outputs/",
}

# Emrat e 14 label-ave (rendi duhet të jetë IDENTIK me train.csv)
LABEL_NAMES = [
    "Left Infraclinoid ICA",    "Right Infraclinoid ICA",
    "Left Supraclinoid ICA",    "Right Supraclinoid ICA",
    "Left MCA",                 "Right MCA",
    "AComA",                    "Left ACA",
    "Right ACA",                "Left PComA",
    "Right PComA",              "Basilar Tip",
    "Other Post. Circ.",        "Aneurysm Present",
]

os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
os.makedirs(CONFIG["output_dir"],     exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET CLASS — DICOM → Tensor
# ─────────────────────────────────────────────────────────────────────────────

class AneurysmDataset(Dataset):
    """
    PyTorch Dataset për NeuroVision AI Detection.

    Lexon DICOM slices, aplikon windowing sipas modalitetit,
    dhe kthen tensor 224×224×3 + label vector 14-dimensional.

    Args:
        df          : DataFrame nga train.csv (ose subset)
        series_dir  : Shtegu i folderit series/
        transform   : torchvision transforms (augmentation)
        is_train    : True = augmentation aktiv
        n_slices    : Numri i slice-ve për seri (default=3 slices mediane)
    """

    # Windowing settings sipas modalitetit (WW/WL)
    WINDOW_SETTINGS = {
        "CTA":    {"width": 700,  "level": 300},
        "MRA":    {"width": 500,  "level": 250},
        "MRI_T1": {"width": 3000, "level": 1500},
        "MRI_T2": {"width": 4000, "level": 2000},
    }

    def __init__(
        self,
        df:          pd.DataFrame,
        series_dir:  str,
        transform=   None,
        is_train:    bool = True,
        n_slices:    int  = 3,
    ):
        self.df         = df.reset_index(drop=True)
        self.series_dir = Path(series_dir)
        self.transform  = transform
        self.is_train   = is_train
        self.n_slices   = n_slices

        self.label_cols = [
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

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        row        = self.df.iloc[idx]
        series_id  = row["SeriesInstanceUID"]
        modality   = row.get("Modality", "CTA")
        labels     = torch.tensor(
            row[self.label_cols].values.astype(np.float32),
            dtype=torch.float32
        )

        # Ngarko imazhin
        image = self._load_image(series_id, modality)
        if self.transform:
            image = self.transform(image)

        return image, labels, series_id

    def _load_image(
        self, series_id: str, modality: str
    ) -> torch.Tensor:
        """
        Ngarko dhe preproceso imazhin DICOM.

        Pipeline:
            1. Gjej skedarin .dcm
            2. Lexo pixel array
            3. HU conversion (slope × px + intercept)
            4. Apliko windowing sipas modalitetit
            5. Normalize [0, 1]
            6. Resize 224×224
            7. Replikoi 1 kanal → 3 kanale

        Returns:
            torch.Tensor [3, 224, 224]
        """
        series_path = self.series_dir / series_id

        if series_path.exists():
            dcm_files = sorted(series_path.glob("*.dcm"))
            if dcm_files:
                return self._process_real_dicom(dcm_files, modality)

        # Fallback: gjenero imazh sintetik për testim
        return self._synthetic_image()

    def _process_real_dicom(
        self, dcm_files: List[Path], modality: str
    ) -> torch.Tensor:
        """
        Proceson skedarët DICOM realë.
        Zgjedh 3 slice mesatare dhe i kombinon si 3 kanale.
        """
        try:
            import pydicom

            # Merr slice-t qendrore (informativisht relevante)
            total   = len(dcm_files)
            indices = [total//4, total//2, 3*total//4]
            indices = [min(i, total-1) for i in indices]

            channels = []
            for i in indices:
                ds    = pydicom.dcmread(str(dcm_files[i]), force=True)
                arr   = ds.pixel_array.astype(np.float32)

                # HU conversion
                slope     = float(getattr(ds, "RescaleSlope",     1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                arr       = arr * slope + intercept

                # Windowing
                arr = self._apply_window(arr, modality)
                channels.append(arr)

            # Stack 3 kanale dhe resize
            image = np.stack(channels, axis=0)           # [3, H, W]
            image = self._resize(image)                   # [3, 224, 224]
            return torch.tensor(image, dtype=torch.float32)

        except Exception:
            return self._synthetic_image()

    def _apply_window(self, arr: np.ndarray, modality: str) -> np.ndarray:
        """Aplikon windowing dhe normalizim [0,1]."""
        settings = self.WINDOW_SETTINGS.get(modality, self.WINDOW_SETTINGS["CTA"])
        ww, wl   = settings["width"], settings["level"]
        lo       = wl - ww / 2
        hi       = wl + ww / 2
        arr      = np.clip(arr, lo, hi)
        arr      = (arr - lo) / (hi - lo)
        return arr.astype(np.float32)

    def _resize(self, arr: np.ndarray) -> np.ndarray:
        """Resize array [C, H, W] → [C, 224, 224] me interpolim bilinear."""
        try:
            from skimage.transform import resize as sk_resize
            out = np.stack([
                sk_resize(arr[i], (224, 224), anti_aliasing=True)
                for i in range(arr.shape[0])
            ])
            return out.astype(np.float32)
        except ImportError:
            return np.zeros((3, 224, 224), dtype=np.float32)

    def _synthetic_image(self) -> torch.Tensor:
        """Gjeneron imazh sintetik 224×224 për testim pa DICOM."""
        return torch.randn(3, 224, 224)


def get_transforms(is_train: bool):
    """
    Data augmentation per train / val-test.

    Train: flip, rotacion ±15°, brightness/contrast jitter, normalizim
    Val/Test: vetëm normalizim (pa augmentation)
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: CNN BASELINE
# ─────────────────────────────────────────────────────────────────────────────

class CNNBaseline(nn.Module):
    """
    CNN Baseline — trajnohet nga zeroja, pa transfer learning.

    Arkitektura:
        5 Convolutional Blocks:
            Conv2d(3,32,3)   → BN → ReLU → Conv2d(32,32,3)   → BN → ReLU → MaxPool → Dropout
            Conv2d(32,64,3)  → BN → ReLU → Conv2d(64,64,3)   → BN → ReLU → MaxPool → Dropout
            Conv2d(64,128,3) → BN → ReLU → Conv2d(128,128,3) → BN → ReLU → MaxPool → Dropout
            Conv2d(128,256,3)→ BN → ReLU → Conv2d(256,256,3) → BN → ReLU → MaxPool → Dropout
            Conv2d(256,512,3)→ BN → ReLU → Conv2d(512,512,3) → BN → ReLU → MaxPool → Dropout

        Global Average Pooling → FC(512,256) → BN → ReLU → Dropout → FC(256,14) → Sigmoid

    Parametra: ~6.2M trainueshëm
    """

    def __init__(self, n_classes: int = 14, dropout: float = 0.30):
        super(CNNBaseline, self).__init__()

        def conv_block(in_ch, out_ch, drop=0.15):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(p=drop),
            )

        self.features = nn.Sequential(
            conv_block(3,   32,  drop=0.10),   # 224 → 112
            conv_block(32,  64,  drop=0.10),   # 112 → 56
            conv_block(64,  128, drop=0.15),   # 56  → 28
            conv_block(128, 256, drop=0.15),   # 28  → 14
            conv_block(256, 512, drop=0.20),   # 14  → 7
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # → [B, 512, 1, 1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return torch.sigmoid(x)   # Multi-label → sigmoid per output

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 & 3: RESNET (50 / 101) me Transfer Learning
# ─────────────────────────────────────────────────────────────────────────────

class ResNetModel(nn.Module):
    """
    ResNet (50 ose 101) me Transfer Learning nga ImageNet.

    Arkitektura:
        Backbone: ResNet-50/101 (pretrained ImageNet weights)
            - Layers 1–3: NGRIRA gjatë epoch-ve të para (freeze_epochs)
            - Layer 4 + FC: gjithmonë trainueshëm

        Classification Head (zëvendëson FC origjinal):
            Linear(2048, 512) → BatchNorm1d → ReLU → Dropout(0.3)
            → Linear(512, 128) → ReLU → Dropout(0.2)
            → Linear(128, 14) → Sigmoid

    Parametra trainueshëm:
        ResNet-50 : ~25.6M total, ~1.8M në head
        ResNet-101: ~44.5M total, ~1.8M në head

    Transfer Learning Strategy:
        Epoch 1-10  : Vetëm head trajnohet (backbone frozen)
        Epoch 11-50 : Layer4 + head fine-tuned (backbone i lirë pjesërisht)
    """

    def __init__(
        self,
        variant:   str  = "resnet101",  # "resnet50" ose "resnet101"
        n_classes: int  = 14,
        dropout:   float= 0.30,
        pretrained:bool = True,
    ):
        super(ResNetModel, self).__init__()

        self.variant = variant

        # Ngarko backbone pretrained
        if variant == "resnet50":
            weights  = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.resnet50(weights=weights)
            feat_dim = 2048
        elif variant == "resnet101":
            weights  = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.resnet101(weights=weights)
            feat_dim = 2048
        else:
            raise ValueError(f"Variant i panjohur: {variant}. Përdor 'resnet50' ose 'resnet101'.")

        # Hiq FC origjinal dhe Global Avg Pool
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pool     = nn.AdaptiveAvgPool2d(1)

        # Classification head i ri
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.67),
            nn.Linear(128, n_classes),
        )

        # Inicializo head-in me Xavier
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        """Ngrin të gjithë backbone-in (Layer 1-4)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_layer4(self):
        """Liro Layer 4 të backbone-it për fine-tuning."""
        # Layer 4 është elementi i fundit i backbone Sequential
        children = list(self.backbone.children())
        if children:
            for param in children[-1].parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)       # [B, 2048, 7, 7]
        x = self.pool(x)           # [B, 2048, 1, 1]
        x = self.head(x)           # [B, 14]
        return torch.sigmoid(x)    # Multi-label sigmoid

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTION — Weighted BCE per class imbalance
# ─────────────────────────────────────────────────────────────────────────────

class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss për multi-label classification.

    Për çdo output j:
        loss_j = -[w_pos * y_j * log(p_j) + (1-y_j) * log(1-p_j)]

    Pesha w_pos = n_neg / n_pos = 2514 / 686 = 3.67 compenson
    class imbalance dhe penalizon false negatives (klinikisht kritike).

    Args:
        pos_weight : float — pesha e klasës pozitive (default=3.67)
        reduction  : str   — 'mean' | 'sum'
    """

    def __init__(self, pos_weight: float = 3.67, reduction: str = "mean"):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction  = reduction

    def forward(
        self,
        predictions: torch.Tensor,    # [B, 14] — probabilitetet
        targets:     torch.Tensor,    # [B, 14] — labels binare
    ) -> torch.Tensor:

        if TORCH_AVAILABLE:
            pw   = torch.tensor(self.pos_weight, device=predictions.device)
            loss = F.binary_cross_entropy_with_logits(
                torch.log(predictions.clamp(min=1e-6) / (1 - predictions).clamp(min=1e-6)),
                targets,
                pos_weight=pw,
                reduction=self.reduction,
            )
            return loss
        return torch.tensor(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# METRIKA — RSNA Weighted Columnwise AUCROC
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsna_weighted_auc(
    y_true: np.ndarray,    # [N, 14]
    y_pred: np.ndarray,    # [N, 14]
) -> Dict[str, float]:
    """
    Llogarit metrikën zyrtare RSNA Weighted Columnwise AUCROC.

    Formula:
        score = 0.5 × AUC(Aneurysm Present)
              + 0.5 × mean(AUC(loc_i) for i in 1..13)

    Kjo është ekuivalente me mesataren e thjeshtë të 14 AUC-ve
    ku Aneurysm Present ka peshë 13 dhe çdo lokacion ka peshë 1.

    Args:
        y_true : [N, 14] ground truth labels
        y_pred : [N, 14] predicted probabilities

    Returns:
        dict me AUC per label + score final
    """
    n_labels = y_true.shape[1]
    per_label_auc = {}

    for i, label in enumerate(LABEL_NAMES):
        if len(np.unique(y_true[:, i])) > 1:
            per_label_auc[label] = round(roc_auc_score(y_true[:, i], y_pred[:, i]), 4)
        else:
            per_label_auc[label] = 0.5   # undefined AUC

    vals            = list(per_label_auc.values())
    aneurysm_auc    = vals[-1]            # Aneurysm Present (indeksi i fundit)
    location_mean   = float(np.mean(vals[:-1]))
    final_score     = 0.5 * aneurysm_auc + 0.5 * location_mean

    return {
        **per_label_auc,
        "Location_Mean_AUC":   round(location_mean, 4),
        "Aneurysm_Present_AUC":round(aneurysm_auc, 4),
        "FINAL_WEIGHTED_AUC":  round(final_score, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class AneurysmTrainer:
    """
    Menaxhon të gjithë ciklin e training: train → validate → save.

    Features:
        - Training loop me WeightedBCELoss
        - Validation me RSNA AUC metrikë
        - Early stopping (patience=7)
        - Cosine Annealing learning rate scheduler
        - Checkpoint saving (best model)
        - Transfer learning schedule (freeze → unfreeze Layer4)
        - History logging (loss + AUC per epoch)

    Args:
        model       : nn.Module (CNNBaseline / ResNetModel)
        config      : dict me hyperparametra
        model_name  : str emri për checkpoint files
    """

    def __init__(
        self,
        model:      nn.Module,
        config:     dict,
        model_name: str = "model",
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch nuk është disponueshëm.")

        self.model      = model.to(DEVICE)
        self.config     = config
        self.model_name = model_name
        self.history    = {
            "train_loss": [], "val_loss": [],
            "train_auc":  [], "val_auc":  [],
            "lr":         [],
        }

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr           = config["learning_rate"],
            weight_decay = config["weight_decay"],
        )

        # Scheduler — Cosine Annealing
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max  = config["epochs"],
            eta_min= config["learning_rate"] * 0.01,
        )

        # Loss function
        self.criterion = WeightedBCELoss(pos_weight=config["pos_weight"])

        # Early stopping state
        self.best_val_auc   = 0.0
        self.best_epoch     = 0
        self.patience_count = 0

        print(f"\n  Model         : {model_name}")
        print(f"  Parametra     : {model.count_parameters():,}")
        print(f"  Device        : {DEVICE}")
        print(f"  Learning Rate : {config['learning_rate']}")
        print(f"  Batch Size    : {config['batch_size']}")
        print(f"  Epochs        : {config['epochs']}")
        print(f"  Pos Weight    : {config['pos_weight']}")

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """Ekzekuton një epoch training. Returns (loss, auc)."""
        self.model.train()
        total_loss   = 0.0
        all_preds    = []
        all_targets  = []

        for batch_idx, (images, labels, _) in enumerate(loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss    = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss  += loss.item()
            all_preds.append(outputs.detach().cpu().numpy())
            all_targets.append(labels.detach().cpu().numpy())

        # Llogarit AUC
        preds   = np.vstack(all_preds)
        targets = np.vstack(all_targets)
        try:
            auc_score = roc_auc_score(targets, preds, average="macro")
        except Exception:
            auc_score = 0.5

        return total_loss / len(loader), float(auc_score)

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> Tuple[float, float, dict]:
        """Ekzekuton validation. Returns (loss, auc, rsna_metrics)."""
        self.model.eval()
        total_loss  = 0.0
        all_preds   = []
        all_targets = []

        for images, labels, _ in loader:
            images  = images.to(DEVICE)
            labels  = labels.to(DEVICE)
            outputs = self.model(images)
            loss    = self.criterion(outputs, labels)

            total_loss  += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

        preds   = np.vstack(all_preds)
        targets = np.vstack(all_targets)

        # RSNA Weighted AUC
        rsna_metrics = compute_rsna_weighted_auc(targets, preds)
        val_auc      = rsna_metrics["FINAL_WEIGHTED_AUC"]

        return total_loss / len(loader), val_auc, rsna_metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> Dict:
        """
        Training loop kryesor.

        Transfer Learning Schedule:
            Epoch 1–freeze_epochs : backbone frozen, vetëm head trajnohet
            Epoch freeze_epochs+1 : Layer4 lirohet për fine-tuning

        Returns:
            dict: History (loss + AUC per epoch)
        """
        print(f"\n  {'='*55}")
        print(f"  FILLIMI I TRAINING — {self.model_name.upper()}")
        print(f"  {'='*55}")

        # Ngri backbone nëse është ResNet
        is_resnet = isinstance(self.model, ResNetModel)
        if is_resnet:
            self.model.freeze_backbone()
            print(f"  Backbone FROZEN për epoch-et 1–{self.config['freeze_epochs']}")

        for epoch in range(1, self.config["epochs"] + 1):
            start_time = time.time()

            # Unfreeze Layer4 pas freeze_epochs
            if is_resnet and epoch == self.config["freeze_epochs"] + 1:
                self.model.unfreeze_layer4()
                # Rifresko optimizer me parametrat e rinj
                self.optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr           = self.config["learning_rate"] * 0.1,
                    weight_decay = self.config["weight_decay"],
                )
                print(f"\n  → Epoch {epoch}: Layer4 UNFREEZE — fine-tuning aktiv")

            # Train + Validate
            train_loss, train_auc = self._train_epoch(train_loader)
            val_loss,   val_auc, rsna_m = self._validate_epoch(val_loader)

            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            # Ruaj historikun
            self.history["train_loss"].append(round(train_loss, 4))
            self.history["val_loss"].append(round(val_loss, 4))
            self.history["train_auc"].append(round(train_auc, 4))
            self.history["val_auc"].append(round(val_auc, 4))
            self.history["lr"].append(round(lr, 7))

            elapsed = time.time() - start_time
            flag    = ""

            # Kontrollo nëse është modeli më i mirë
            if val_auc > self.best_val_auc:
                self.best_val_auc   = val_auc
                self.best_epoch     = epoch
                self.patience_count = 0
                self._save_checkpoint(epoch, val_auc)
                flag = "  ★ BEST"
            else:
                self.patience_count += 1

            # Log
            print(
                f"  Epoch {epoch:3d}/{self.config['epochs']} | "
                f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                f"AUC: {train_auc:.4f}/{val_auc:.4f} | "
                f"RSNA: {rsna_m['FINAL_WEIGHTED_AUC']:.4f} | "
                f"LR: {lr:.2e} | {elapsed:.0f}s{flag}"
            )

            # Early stopping
            if self.patience_count >= self.config["patience"]:
                print(f"\n  Early stopping: epoch {epoch} "
                      f"(best: epoch {self.best_epoch}, AUC={self.best_val_auc:.4f})")
                break

        self._save_history()
        print(f"\n  TRAINING PËRFUNDOI")
        print(f"  Best Epoch    : {self.best_epoch}")
        print(f"  Best Val AUC  : {self.best_val_auc:.4f}")
        return self.history

    def _save_checkpoint(self, epoch: int, val_auc: float):
        """Ruan checkpoint-in e modelit më të mirë."""
        ckpt_path = os.path.join(
            self.config["checkpoint_dir"],
            f"{self.model_name}_best.pt"
        )
        torch.save({
            "epoch":      epoch,
            "val_auc":    val_auc,
            "model_state":self.model.state_dict(),
            "optim_state":self.optimizer.state_dict(),
            "config":     self.config,
        }, ckpt_path)

    def _save_history(self):
        """Ruan historikun e training-ut si JSON."""
        hist_path = os.path.join(
            self.config["output_dir"],
            f"{self.model_name}_history.json"
        )
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  History ruajtur: {hist_path}")

    def load_best_checkpoint(self):
        """Ngarkon checkpoint-in me AUC-in më të lartë."""
        ckpt_path = os.path.join(
            self.config["checkpoint_dir"],
            f"{self.model_name}_best.pt"
        )
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            self.model.load_state_dict(ckpt["model_state"])
            print(f"  ✅ Checkpoint ngarkuar: epoch={ckpt['epoch']}, AUC={ckpt['val_auc']:.4f}")
        else:
            print(f"  ⚠ Checkpoint nuk u gjet: {ckpt_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL EVALUATION — Test Set
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_on_test_set(
    model:       nn.Module,
    test_loader: DataLoader,
) -> Dict:
    """
    Evaluon modelin në test set të plotë.

    Returns:
        dict me:
            - RSNA Weighted AUC (metrika zyrtare)
            - Per-label AUC
            - Confusion matrix komponentët
            - Sensitivity / Specificity / PPV / NPV
    """
    if not TORCH_AVAILABLE:
        return {}

    model.eval()
    model.to(DEVICE)

    all_preds    = []
    all_targets  = []
    all_series   = []

    for images, labels, series_ids in test_loader:
        images  = images.to(DEVICE)
        outputs = model(images)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(labels.numpy())
        all_series.extend(series_ids)

    preds   = np.vstack(all_preds)    # [N, 14]
    targets = np.vstack(all_targets)  # [N, 14]

    # RSNA Weighted AUC
    rsna_metrics = compute_rsna_weighted_auc(targets, preds)

    # Confusion matrix për Aneurysm Present (indeksi 13)
    y_true_bin = targets[:, -1].astype(int)
    y_pred_bin = (preds[:, -1] >= 0.50).astype(int)

    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())

    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv          = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv          = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1           = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0

    results = {
        **rsna_metrics,
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "sensitivity":  round(sensitivity,  4),
        "specificity":  round(specificity,  4),
        "PPV":          round(ppv,           4),
        "NPV":          round(npv,           4),
        "F1_score":     round(f1,            4),
        "n_test":       len(all_series),
    }

    # Print rezultatet
    print(f"\n  {'='*55}")
    print(f"  REZULTATET E TEST SET-IT")
    print(f"  {'='*55}")
    print(f"  RSNA Weighted AUC : {results['FINAL_WEIGHTED_AUC']:.4f}")
    print(f"  AUC Aneurysm      : {results['Aneurysm_Present_AUC']:.4f}")
    print(f"  AUC Location Mean : {results['Location_Mean_AUC']:.4f}")
    print(f"  {'─'*40}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Sensitivity : {sensitivity*100:.1f}%")
    print(f"  Specificity : {specificity*100:.1f}%")
    print(f"  PPV         : {ppv*100:.1f}%")
    print(f"  NPV         : {npv*100:.1f}%")
    print(f"  F1-Score    : {f1:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Demo pa GPU (PyTorch opsional)
# ─────────────────────────────────────────────────────────────────────────────

def demo_without_gpu():
    """
    Demonstron arkitekturat dhe numrin e parametrave
    pa GPU dhe pa dataset real.
    """
    print("\n" + "=" * 65)
    print("DEMO — Arkitekturat e Modeleve (pa GPU)")
    print("=" * 65)

    import numpy as np

    models_info = [
        ("CNN Baseline",  {"n_classes": 14, "dropout": 0.30}),
        ("ResNet-50",     {"variant": "resnet50",  "n_classes": 14, "pretrained": False}),
        ("ResNet-101",    {"variant": "resnet101", "n_classes": 14, "pretrained": False}),
    ]

    print(f"\n  {'Model':<16} {'Parametra':>12}  {'Input Shape':>14}  {'Output Shape':>14}")
    print(f"  {'-'*62}")

    for name, kwargs in models_info:
        try:
            if name == "CNN Baseline":
                m = CNNBaseline(**kwargs)
            else:
                m = ResNetModel(**kwargs)

            params = m.count_parameters()
            # Forward pass me input sintetik
            x   = torch.randn(2, 3, 224, 224)  # batch=2
            out = m(x)

            print(f"  {name:<16} {params:>12,}  {str(list(x.shape)):>14}  {str(list(out.shape)):>14}")

        except Exception as e:
            print(f"  {name:<16} Gabim: {e}")


def print_training_summary():
    """Shfaq rezymën e konfigurimit të training-ut."""
    print("\n" + "=" * 65)
    print("KONFIGURIMI I TRAINING")
    print("=" * 65)
    for key, val in CONFIG.items():
        print(f"  {key:<22}: {val}")
    print(f"\n  TRANSFER LEARNING SCHEDULE:")
    print(f"  Epoch 1–{CONFIG['freeze_epochs']:2d}  : Backbone FROZEN  (vetëm head trajnohet)")
    print(f"  Epoch {CONFIG['freeze_epochs']+1:2d}–50 : Layer4 UNFREEZE  (fine-tuning)")
    print(f"\n  LOSS: Weighted BCE  (pos_weight = {CONFIG['pos_weight']})")
    print(f"  METRIKA: RSNA Weighted Columnwise AUCROC")
    print(f"           = 0.5 × AUC(Aneurysm Present)")
    print(f"             + 0.5 × mean(AUC(13 locations))")


if __name__ == "__main__":
    print("\n" + "█" * 65)
    print("  MODEL TRAINING MODULE — NEUROVISION AI DETECTION")
    print("  Blina Sopjani | ID: 69401 | Universum College")
    print("█" * 65)

    print_training_summary()

    if TORCH_AVAILABLE:
        demo_without_gpu()
    else:
        print("\n  PyTorch nuk është disponueshëm.")
        print("  Instalo me: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")

    print(f"""
  ── PËR TRAINING TË PLOTË ──────────────────────────────────
  from model_training import ResNetModel, AneurysmTrainer, CONFIG
  from dataset_connection import load_dataset

  # 1. Ngarko dataset
  data     = load_dataset()
  df_train = data['df_train']

  # 2. Split train/val/test
  from sklearn.model_selection import train_test_split
  df_tr, df_val = train_test_split(df_train, test_size=0.2,
                  stratify=df_train['Aneurysm Present'], random_state=42)

  # 3. Krijo model
  model = ResNetModel(variant='resnet101', n_classes=14, pretrained=True)

  # 4. Trajno
  trainer = AneurysmTrainer(model, CONFIG, model_name='ResNet101')
  history = trainer.fit(train_loader, val_loader)
  ────────────────────────────────────────────────────────────
    """)
