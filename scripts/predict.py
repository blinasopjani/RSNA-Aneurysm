"""
=============================================================================
NeuroVision AI — INFERENCE / PREDICTION MODULE
=============================================================================
Student   : Blina Sopjani | ID: 69401
File      : predict.py

Ky file implementon funksionin predict() në formatin RSNA zyrtar,
bazuar saktë në RSNA_Aneurysm_Detection_Demo.txt.

FORMATI ZYRTAR RSNA:
    def predict(series_path: str) -> pl.DataFrame | pd.DataFrame:
        - Input : shtegu i folderit të serisë DICOM
        - Output: DataFrame me SeriesInstanceUID + 14 kolona probabiliteti

PIPELINE:
    1. Lexo skedarët .dcm nga series_path
    2. Preproceso me pipeline (windowing → resize → normalize)
    3. Ngarko modelin ResNet-101 nga checkpoint
    4. Bëj inference → 14 probabilitete
    5. Kthen DataFrame në formatin RSNA

REQUIREMENTS:
    pip install torch torchvision pydicom pandas numpy
=============================================================================
"""

import os
import gc
import shutil
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Optional

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS — Torch dhe modulet lokale
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    from torchvision import models
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# KONSTANTET — saktë nga dokumentacioni RSNA
# ─────────────────────────────────────────────────────────────────────────────

ID_COL = "SeriesInstanceUID"

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
    "Aneurysm Present",   # Target kryesor — weight=13 në metrikën RSNA
]

# DICOM tags të lejuara (saktë nga RSNA_Aneurysm_Detection_Demo.txt)
DICOM_TAG_ALLOWLIST = [
    "BitsAllocated", "BitsStored", "Columns", "FrameOfReferenceUID",
    "HighBit", "ImageOrientationPatient", "ImagePositionPatient",
    "InstanceNumber", "Modality", "PatientID", "PhotometricInterpretation",
    "PixelRepresentation", "PixelSpacing", "PlanarConfiguration",
    "RescaleIntercept", "RescaleSlope", "RescaleType", "Rows",
    "SOPClassUID", "SOPInstanceUID", "SamplesPerPixel",
    "SliceThickness", "SpacingBetweenSlices", "StudyInstanceUID",
    "TransferSyntaxUID",
]

WINDOW_SETTINGS = {
    "CTA":    {"ww": 700,  "wl": 300},
    "MRA":    {"ww": 500,  "wl": 250},
    "MRI_T1": {"ww": 3000, "wl": 1500},
    "MRI_T2": {"ww": 4000, "wl": 2000},
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CHECKPOINT_PATH = "./checkpoints/ResNet101_best.pt"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER — singleton (ngarkohet vetëm 1 herë)
# ─────────────────────────────────────────────────────────────────────────────

_model_cache = None  # type: ignore

def get_model():
    """
    Ngarkon modelin ResNet-101 nga checkpoint.
    Singleton — modeli ngarkohet vetëm herën e parë që thirret.

    Sipas udhëzimeve RSNA:
        "Nëse nevojiten më shumë se 15 min për të ngarkuar modelin,
         mund ta bësh gjatë thirrjes së parë të predict()."

    Returns:
        nn.Module ose None nëse checkpoint nuk ekziston
    """
    global _model_cache

    if _model_cache is not None:
        return _model_cache

    if not TORCH_AVAILABLE:
        return None

    try:
        # Krijo arkitekturën ResNet-101
        backbone = models.resnet101(weights=None)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        pool     = nn.AdaptiveAvgPool2d(1)

        head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, len(LABEL_COLS)),
        )

        class ResNet101Inference(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.pool     = pool
                self.head     = head

            def forward(self, x):
                x = self.backbone(x)
                x = self.pool(x)
                x = self.head(x)
                return torch.sigmoid(x)

        model = ResNet101Inference()

        # Ngarko weights nëse checkpoint ekziston
        if os.path.exists(CHECKPOINT_PATH):
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state"])
            print(f"  ✅ Checkpoint ngarkuar: epoch={ckpt.get('epoch','?')}, "
                  f"AUC={ckpt.get('val_auc', '?')}")
        else:
            print(f"  ⚠ Checkpoint nuk u gjet: {CHECKPOINT_PATH}")
            print(f"     Duke përdorur peshë random (vetëm për demo).")

        model.eval()
        model.to(DEVICE)
        _model_cache = model
        return model

    except Exception as e:
        print(f"  ❌ Gabim në ngarkimin e modelit: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING HELPERS (standalone, pa varësi nga preprocessing.py)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_window(arr: np.ndarray, modality: str) -> np.ndarray:
    """Windowing + normalizim [0,1]."""
    s  = WINDOW_SETTINGS.get(modality, WINDOW_SETTINGS["CTA"])
    lo = s["wl"] - s["ww"] / 2
    hi = s["wl"] + s["ww"] / 2
    return ((np.clip(arr, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def _resize_224(arr: np.ndarray) -> np.ndarray:
    """Resize [C,H,W] → [C,224,224]."""
    try:
        from skimage.transform import resize as sk_resize
        return np.stack([
            sk_resize(arr[i], (224, 224), anti_aliasing=True,
                      preserve_range=True).astype(np.float32)
            for i in range(arr.shape[0])
        ])
    except ImportError:
        pass
    try:
        import cv2
        return np.stack([
            cv2.resize(arr[i], (224, 224), interpolation=cv2.INTER_LINEAR)
            for i in range(arr.shape[0])
        ])
    except ImportError:
        return np.zeros((arr.shape[0], 224, 224), dtype=np.float32)


def _normalize(arr: np.ndarray) -> np.ndarray:
    """ImageNet normalizim per 3 kanalet e para."""
    out = arr.copy()
    for i in range(min(3, arr.shape[0])):
        out[i] = (arr[i] - IMAGENET_MEAN[i]) / (IMAGENET_STD[i] + 1e-8)
    return out.astype(np.float32)


def _read_and_preprocess(series_path: str) -> np.ndarray:
    """
    Lexo DICOM skedarët dhe kthen tensor [3, 224, 224].

    Pipeline:
        1. Lexo .dcm files → pixel arrays (HU)
        2. Rendit sipas ImagePositionPatient Z
        3. Merr 3 slice-t qendrore (Q25, Q50, Q75)
        4. Multi-window CTA ose single window per MRI/MRA
        5. Resize 224×224
        6. ImageNet normalizim

    Returns:
        np.ndarray [3, 224, 224] gati për model
    """
    try:
        import pydicom
    except ImportError:
        return np.zeros((3, 224, 224), dtype=np.float32)

    dcm_files = sorted(Path(series_path).glob("*.dcm"))
    if not dcm_files:
        return np.zeros((3, 224, 224), dtype=np.float32)

    # ── Lexo të gjitha slice-t ────────────────────────────────────────────
    slices = []
    modality = "CTA"
    for dcm_path in dcm_files:
        try:
            ds  = pydicom.dcmread(str(dcm_path), force=True)
            arr = ds.pixel_array.astype(np.float32)

            slope     = float(getattr(ds, "RescaleSlope",     1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            hu        = arr * slope + intercept

            ipp = getattr(ds, "ImagePositionPatient", [0, 0, 0])
            z   = float(ipp[2]) if hasattr(ipp, "__len__") else 0.0
            modality = str(getattr(ds, "Modality", "CTA"))

            slices.append({"z": z, "hu": hu})
        except Exception:
            continue

    if not slices:
        return np.zeros((3, 224, 224), dtype=np.float32)

    # ── Rendit sipas Z dhe merr 3 slice-t qendrore ────────────────────────
    slices.sort(key=lambda s: s["z"])
    total   = len(slices)
    indices = [total // 4, total // 2, 3 * total // 4]
    indices = [min(i, total - 1) for i in indices]

    # ── Windowing + stack ─────────────────────────────────────────────────
    channels = []
    for idx in indices:
        win = _apply_window(slices[idx]["hu"], modality)  # [H, W]
        channels.append(win)

    arr_3ch = np.stack(channels, axis=0)   # [3, H, W]

    # ── Resize + Normalizim ───────────────────────────────────────────────
    arr_3ch = _resize_224(arr_3ch)         # [3, 224, 224]
    arr_3ch = _normalize(arr_3ch)          # [3, 224, 224] ImageNet norm

    return arr_3ch.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT FUNCTION — FORMATI ZYRTAR RSNA
# ─────────────────────────────────────────────────────────────────────────────

def predict(series_path: str):
    """
    Funksioni kryesor i inferencing — formati zyrtar RSNA Competition.

    Sipas RSNA_Aneurysm_Detection_Demo.txt:
        - Input : series_path (shtegu i folderit të serisë DICOM)
        - Output: DataFrame me [SeriesInstanceUID + 14 kolona probabiliteti]
        - Çdo parashikim (veç të parit) duhet kthyer brenda 30 minutave
        - Modeli mund të ngarkohet gjatë thirrjes së parë

    Kolona e output-it:
        SeriesInstanceUID, Left Infraclinoid ICA, ..., Aneurysm Present
        (14 kolona probabiliteti [0.0, 1.0])

    Args:
        series_path: str — shtegu i folderit series/{SeriesInstanceUID}/

    Returns:
        pl.DataFrame | pd.DataFrame — 1 rresht me 14 probabilitete
    """

    # ── Nxjerr SeriesInstanceUID nga shtegu ──────────────────────────────
    series_id = os.path.basename(series_path.rstrip("/\\"))

    # ── Mbledh DICOM tags (saktë si në demo zyrtar RSNA) ─────────────────
    tags = defaultdict(list)
    tags[ID_COL] = series_id

    all_filepaths = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith(".dcm"):
                all_filepaths.append(os.path.join(root, file))
    all_filepaths.sort()

    try:
        import pydicom
        for filepath in all_filepaths:
            ds = pydicom.dcmread(filepath, force=True)
            tags["filepath"].append(filepath)
            for tag in DICOM_TAG_ALLOWLIST:
                tags[tag].append(getattr(ds, tag, None))
    except ImportError:
        pass

    # ── Preproceso imazhin ────────────────────────────────────────────────
    image_array = _read_and_preprocess(series_path)    # [3, 224, 224]

    # ── Inference ─────────────────────────────────────────────────────────
    model = get_model()

    if model is not None and TORCH_AVAILABLE:
        try:
            with torch.no_grad():
                tensor = torch.tensor(image_array).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]
                output = model(tensor)                                        # [1, 14]
                probs  = output.squeeze(0).cpu().numpy()                     # [14]
        except Exception as e:
            print(f"  ⚠ Inference error: {e} — duke kthyer probabilitete default")
            probs = np.full(len(LABEL_COLS), 0.5, dtype=np.float32)
    else:
        # Fallback nëse modeli nuk është disponueshëm
        probs = np.full(len(LABEL_COLS), 0.5, dtype=np.float32)

    # Siguroji që probabilitetet janë [0.0, 1.0]
    probs = np.clip(probs, 0.0, 1.0).astype(float)

    # ── Krijo DataFrame në formatin RSNA ─────────────────────────────────
    if POLARS_AVAILABLE:
        predictions = pl.DataFrame(
            data   = [[series_id] + probs.tolist()],
            schema = [ID_COL, *LABEL_COLS],
            orient = "row",
        )
    else:
        predictions = pd.DataFrame(
            data    = [[series_id] + probs.tolist()],
            columns = [ID_COL] + LABEL_COLS,
        )

    # ── Validim i output-it (saktë si në demo zyrtar) ────────────────────
    if POLARS_AVAILABLE and isinstance(predictions, pl.DataFrame):
        assert predictions.columns == [ID_COL, *LABEL_COLS], \
            f"Kolonat gabim: {predictions.columns}"
    elif isinstance(predictions, pd.DataFrame):
        assert (predictions.columns == [ID_COL, *LABEL_COLS]).all(), \
            f"Kolonat gabim: {predictions.columns.tolist()}"

    # ── KRITIKE: pastro /kaggle/shared (sipas udhëzimeve RSNA) ───────────
    shutil.rmtree("/kaggle/shared", ignore_errors=True)

    # Hiq SeriesInstanceUID nga output (e kthyer veç brenda, jo në submission)
    if POLARS_AVAILABLE and isinstance(predictions, pl.DataFrame):
        return predictions.drop(ID_COL)
    else:
        return predictions.drop(columns=[ID_COL])


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PREDICTOR — për evaluim lokal
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch(
    df:         "pd.DataFrame",
    series_dir: str,
    max_series: Optional[int] = None,
) -> pd.DataFrame:
    """
    Bën inference mbi batch-in e plotë të test set-it.

    Përdoret lokalisht për evaluim — nuk është pjesë e RSNA API-t.

    Args:
        df         : DataFrame nga train.csv (ose test set)
        series_dir : Shtegu i folderit series/
        max_series : Kufizim i numrit (None = të gjitha)

    Returns:
        pd.DataFrame me parashikimet dhe ground truth (nëse disponueshëm)
    """
    series_list = df["SeriesInstanceUID"].tolist()
    if max_series:
        series_list = series_list[:max_series]

    all_preds = []
    total     = len(series_list)

    print(f"\n  Batch inference: {total} serje...")

    for i, sid in enumerate(series_list, 1):
        spath = os.path.join(series_dir, sid)

        try:
            result_df = predict(spath)

            if POLARS_AVAILABLE and hasattr(result_df, "to_pandas"):
                result_pd = result_df.to_pandas()
            else:
                result_pd = result_df if isinstance(result_df, pd.DataFrame) else pd.DataFrame(result_df)

            result_pd[ID_COL] = sid
            all_preds.append(result_pd)

        except Exception as e:
            print(f"  ⚠ Seri {sid}: {e}")
            fallback = pd.DataFrame(
                [[sid] + [0.5] * len(LABEL_COLS)],
                columns=[ID_COL] + LABEL_COLS
            )
            all_preds.append(fallback)

        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] {i/total*100:.0f}%")

        # Pastro memory
        gc.collect()

    result = pd.concat(all_preds, ignore_index=True)

    # Shto ground truth nëse disponueshëm
    if "Aneurysm Present" in df.columns:
        gt_cols     = [ID_COL] + LABEL_COLS
        gt_available = [c for c in gt_cols if c in df.columns]
        result = result.merge(
            df[gt_available].rename(
                columns={c: f"{c}_true" for c in LABEL_COLS if c in gt_available}
            ),
            on=ID_COL, how="left"
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# RSNA INFERENCE SERVER — formati zyrtar për submission
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_server():
    """
    Nis RSNA Inference Server-in.

    Kur notebook ekzekutohet në hidden test set,
    inference_server.serve() duhet thirrur brenda 15 minutave.

    Kur ekzekutohet lokalisht, run_local_gateway() testohet me
    seriet e publikuara.
    """
    try:
        import kaggle_evaluation.rsna_inference_server as rsna_server

        inference_server = rsna_server.RSNAInferenceServer(predict)

        if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            # Submission zyrtare në Kaggle
            print("  → Nis server-in (KAGGLE_IS_COMPETITION_RERUN=True)...")
            inference_server.serve()
        else:
            # Test lokal
            print("  → Test lokal me gateway...")
            inference_server.run_local_gateway()

            # Shfaq submission
            try:
                if POLARS_AVAILABLE:
                    import polars as _pl
                    submission = _pl.read_parquet("/kaggle/working/submission.parquet")
                    print(f"\n  Submission shape: {submission.shape}")
                    print(submission.head(3))
                else:
                    submission = pd.read_parquet("/kaggle/working/submission.parquet")
                    print(f"\n  Submission shape: {submission.shape}")
                    print(submission.head(3))
            except Exception:
                print("  ℹ submission.parquet nuk u gjet (normal në zhvillim lokal)")

    except ImportError:
        print("  ⚠ kaggle_evaluation nuk disponueshëm (vetëm brenda Kaggle Notebook).")
        print("     Lokalisht: përdor predict() direkt ose predict_batch().")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "█" * 65)
    print("  PREDICTION MODULE — NEUROVISION AI DETECTION")
    print("  Blina Sopjani | ID: 69401 | Universum College")
    print("█" * 65)

    print(f"""
  ── FORMATI ZYRTAR RSNA ──────────────────────────────────
  Input  : series_path (shtegu DICOM folder)
  Output : DataFrame [SeriesInstanceUID + 14 label probabilities]

  Kolonat output:
""")
    for i, col in enumerate(LABEL_COLS, 1):
        weight = "(weight=13)" if col == "Aneurysm Present" else "(weight=1) "
        print(f"    {i:2d}. {col:<45} {weight}")

    print(f"""
  ── TEST LOKAL ───────────────────────────────────────────
  from predict import predict

  result = predict('./series/1.2.826.0.1.000000000001/')
  print(result)

  ── RSNA SERVER ──────────────────────────────────────────
  from predict import run_inference_server
  run_inference_server()
  ──────────────────────────────────────────────────────────
    """)

    # Demo me seri sintetike
    print("  Demo me seri sintetike (pa DICOM real):")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        result = predict(tmp)
        print(f"  Output type  : {type(result).__name__}")
        if hasattr(result, "shape"):
            print(f"  Output shape : {result.shape}")
        print(f"  Kolonat      : {list(result.columns)}")
        print(f"  Vlerat sample: {result.values[0][:5].tolist()} ...")
