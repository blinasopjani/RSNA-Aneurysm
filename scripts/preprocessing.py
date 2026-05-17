"""
=============================================================================
NeuroVision AI — DICOM PREPROCESSING PIPELINE
=============================================================================
Student   : Blina Sopjani | ID: 69401
File      : preprocessing.py

PIPELINE:
  Step 1 — DICOM lexim dhe HU conversion
  Step 2 — Windowing sipas modalitetit (CTA/MRA/MRI)
  Step 3 — Normalizim [0, 1]
  Step 4 — Resize 224×224
  Step 5 — 3-channel stack (grayscale → RGB)
  Step 6 — Augmentation (vetëm train set)
  Step 7 — Batch preparation për model input

REQUIREMENTS:
  pip install pydicom numpy scipy scikit-image opencv-python
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional, Dict

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# WINDOWING SETTINGS (WW/WL) sipas modalitetit
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_PRESETS = {
    # Modality       Width   Level   Arsyeja klinike
    "CTA":    {"ww": 700,  "wl": 300},   # Angiography window — struktura vaskulare
    "MRA":    {"ww": 500,  "wl": 250},   # MR Angio — flow-sensitive
    "MRI_T1": {"ww": 3000, "wl": 1500},  # T1 soft tissue contrast
    "MRI_T2": {"ww": 4000, "wl": 2000},  # T2 fluid/CSF visibility
    "DEFAULT":{"ww": 700,  "wl": 300},   # Fallback
}

# Numri i slice-ve për t'u marrë nga çdo seri
N_SLICES_PER_SERIES = 3   # Qendrore, Q1, Q3


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DICOM LEXIM DHE HU CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def read_dicom_series(series_path: str) -> List[Dict]:
    """
    Lexon të gjitha skedarët .dcm nga një folder serje.
    Aplikon HU (Hounsfield Unit) conversion me:
        HU = pixel_value × RescaleSlope + RescaleIntercept

    Args:
        series_path: Shtegu i folderit  series/{SeriesInstanceUID}/

    Returns:
        List[dict] — slice-t e renditura sipas pozicionit Z,
        secili me fushat: z_pos, sop_uid, hu_array, modality
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError("pip install pydicom")

    dcm_files = sorted(Path(series_path).glob("*.dcm"))
    if not dcm_files:
        return []

    slices = []
    for dcm_path in dcm_files:
        try:
            ds  = pydicom.dcmread(str(dcm_path), force=True)
            arr = ds.pixel_array.astype(np.float32)

            # HU conversion
            slope     = float(getattr(ds, "RescaleSlope",     1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            hu_arr    = arr * slope + intercept

            # Pozicioni Z (për renditje kranio-kaudale)
            ipp = getattr(ds, "ImagePositionPatient", [0, 0, 0])
            z   = float(ipp[2]) if hasattr(ipp, "__len__") else float(dcm_path.stem)

            slices.append({
                "z_pos":    z,
                "sop_uid":  str(getattr(ds, "SOPInstanceUID", dcm_path.stem)),
                "hu_array": hu_arr,
                "modality": str(getattr(ds, "Modality", "CTA")),
                "rows":     int(getattr(ds, "Rows",    512)),
                "cols":     int(getattr(ds, "Columns", 512)),
                "filepath": str(dcm_path),
            })
        except Exception:
            continue

    # Rendit sipas Z (kraniokaudal)
    slices.sort(key=lambda s: s["z_pos"])
    return slices


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — WINDOWING
# ─────────────────────────────────────────────────────────────────────────────

def apply_windowing(
    hu_array: np.ndarray,
    modality: str = "CTA",
    custom_ww: Optional[float] = None,
    custom_wl: Optional[float] = None,
) -> np.ndarray:
    """
    Aplikon windowing (contrast adjustment) në imazhin HU.

    Formula:
        lower = WL - WW/2
        upper = WL + WW/2
        output = clip(HU, lower, upper)

    Çdo modalitet ka window optimal:
        CTA    WW=700,  WL=300  → veçon strukturat vaskulare
        MRA    WW=500,  WL=250  → flow-sensitive signal
        MRI_T1 WW=3000, WL=1500 → soft tissue contrast
        MRI_T2 WW=4000, WL=2000 → fluid/CSF

    Args:
        hu_array   : Array 2D me vlera HU
        modality   : "CTA"|"MRA"|"MRI_T1"|"MRI_T2"
        custom_ww  : Window Width (override)
        custom_wl  : Window Level (override)

    Returns:
        np.ndarray [H, W] float32 me vlera [0.0, 1.0]
    """
    preset = WINDOW_PRESETS.get(modality, WINDOW_PRESETS["DEFAULT"])
    ww     = custom_ww if custom_ww is not None else preset["ww"]
    wl     = custom_wl if custom_wl is not None else preset["wl"]

    lower = wl - ww / 2.0
    upper = wl + ww / 2.0

    windowed    = np.clip(hu_array, lower, upper)
    normalized  = (windowed - lower) / (upper - lower)
    return normalized.astype(np.float32)


def apply_multi_window(
    hu_array: np.ndarray,
    modality: str = "CTA",
) -> np.ndarray:
    """
    Gjeneron 3 kanale me window-settings të ndryshme.
    Kjo teknikë kap informacion nga kontraste të shumta njëkohësisht.

    Për CTA: kanal 1=brain, kanal 2=blood/vessel, kanal 3=bone
    Shembull:
        Ch1: WW=80,   WL=40   (brain window)
        Ch2: WW=700,  WL=300  (angio window)
        Ch3: WW=1500, WL=400  (bone window)

    Returns:
        np.ndarray [3, H, W] float32
    """
    if modality in ("CTA", "DEFAULT"):
        ch1 = apply_windowing(hu_array, custom_ww=80,   custom_wl=40)
        ch2 = apply_windowing(hu_array, custom_ww=700,  custom_wl=300)
        ch3 = apply_windowing(hu_array, custom_ww=1500, custom_wl=400)
    elif modality == "MRA":
        ch1 = apply_windowing(hu_array, custom_ww=200,  custom_wl=100)
        ch2 = apply_windowing(hu_array, custom_ww=500,  custom_wl=250)
        ch3 = apply_windowing(hu_array, custom_ww=1000, custom_wl=500)
    else:  # MRI
        ch1 = apply_windowing(hu_array, custom_ww=1000, custom_wl=500)
        ch2 = apply_windowing(hu_array, custom_ww=3000, custom_wl=1500)
        ch3 = apply_windowing(hu_array, custom_ww=5000, custom_wl=2500)

    return np.stack([ch1, ch2, ch3], axis=0)   # [3, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — RESIZE
# ─────────────────────────────────────────────────────────────────────────────

def resize_image(
    arr:         np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Resize imazhin në dimensionin e kërkuar.

    Mbështet:
        - [H, W]     → [H', W']
        - [C, H, W]  → [C, H', W']

    Args:
        arr         : Array 2D ose 3D
        target_size : (height, width) — default (224, 224) për ResNet

    Returns:
        np.ndarray me target_size
    """
    try:
        from skimage.transform import resize as sk_resize

        if arr.ndim == 2:
            return sk_resize(arr, target_size, anti_aliasing=True,
                             preserve_range=True).astype(np.float32)
        elif arr.ndim == 3:  # [C, H, W]
            resized = np.stack([
                sk_resize(arr[i], target_size, anti_aliasing=True,
                          preserve_range=True)
                for i in range(arr.shape[0])
            ])
            return resized.astype(np.float32)

    except ImportError:
        pass

    try:
        import cv2
        if arr.ndim == 2:
            return cv2.resize(arr, (target_size[1], target_size[0]),
                              interpolation=cv2.INTER_LINEAR).astype(np.float32)
        elif arr.ndim == 3:
            resized = np.stack([
                cv2.resize(arr[i], (target_size[1], target_size[0]),
                           interpolation=cv2.INTER_LINEAR)
                for i in range(arr.shape[0])
            ])
            return resized.astype(np.float32)

    except ImportError:
        pass

    # Fallback i thjeshtë me numpy (less accurate)
    if arr.ndim == 2:
        from scipy.ndimage import zoom
        zy = target_size[0] / arr.shape[0]
        zx = target_size[1] / arr.shape[1]
        return zoom(arr, (zy, zx)).astype(np.float32)

    return np.zeros((arr.shape[0], *target_size), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SLICE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def select_slices(
    slices:   List[Dict],
    n:        int = 3,
    strategy: str = "quartile",
) -> List[Dict]:
    """
    Zgjedh n slice-t më informative nga seria DICOM.

    Strategji:
        "quartile"  : Q25, Q50, Q75 — shpërndarje e barabartë
        "central"   : n slice-t qendrore
        "uniform"   : n slice-t me hapësirë uniforme
        "max_signal": slice-t me variancën më të lartë

    Args:
        slices   : Lista e slice-ve të renditura sipas Z
        n        : Numri i slice-ve për t'u zgjedhur
        strategy : Strategjia e zgjedhjes

    Returns:
        List[Dict] — n slice-t e zgjedhura
    """
    total = len(slices)
    if total == 0:
        return []
    if total <= n:
        return slices

    if strategy == "quartile":
        indices = [int(total * q) for q in np.linspace(0.25, 0.75, n)]
    elif strategy == "central":
        mid     = total // 2
        half    = n // 2
        indices = list(range(max(0, mid - half), min(total, mid - half + n)))
    elif strategy == "uniform":
        indices = [int(i) for i in np.linspace(0, total - 1, n)]
    elif strategy == "max_signal":
        variances = [np.var(s["hu_array"]) for s in slices]
        indices   = sorted(np.argsort(variances)[-n:])
    else:
        indices = [int(i) for i in np.linspace(0, total - 1, n)]

    indices = [min(i, total - 1) for i in indices]
    return [slices[i] for i in indices]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — FULL PIPELINE: Series → Tensor
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_series(
    series_path:  str,
    modality:     str = "CTA",
    target_size:  Tuple[int, int] = (224, 224),
    n_slices:     int = 3,
    strategy:     str = "quartile",
    use_multi_window: bool = True,
) -> np.ndarray:
    """
    Pipeline i plotë: folder DICOM → tensor [C×n, H, W] gati për model.

    Hapat:
        1. Lexo skedarët .dcm → HU arrays
        2. Rendit sipas Z (kraniokaudal)
        3. Zgjedh n_slices slice informative
        4. Apliko windowing (single ose multi-window)
        5. Resize çdo slice në target_size
        6. Stack të gjitha slicet si kanale

    Args:
        series_path      : Shtegu i folderit të serisë
        modality         : "CTA"|"MRA"|"MRI_T1"|"MRI_T2"
        target_size      : (H, W) — (224, 224) për ResNet
        n_slices         : Numri i sliceve (default=3)
        strategy         : Strategjia e zgjedhjes së sliceve
        use_multi_window : True = 3 kanale per slice (9 kanale total)
                           False = 1 kanal per slice (3 kanale total)

    Returns:
        np.ndarray [C, 224, 224] float32
            C = n_slices × 3 nëse use_multi_window
            C = n_slices     nëse single window
    """
    # Lexo DICOM
    slices = read_dicom_series(series_path)
    if not slices:
        channels = 9 if use_multi_window else 3
        return np.zeros((channels, *target_size), dtype=np.float32)

    modality = slices[0].get("modality", modality)

    # Zgjedh slicet
    selected = select_slices(slices, n=n_slices, strategy=strategy)

    channels = []
    for sl in selected:
        hu = sl["hu_array"]

        if use_multi_window:
            # 3 kanale per slice (brain + vessel + bone windows)
            ch = apply_multi_window(hu, modality)          # [3, H, W]
        else:
            # 1 kanal per slice, replikuar 3x
            win = apply_windowing(hu, modality)             # [H, W]
            ch  = np.stack([win, win, win], axis=0)         # [3, H, W]

        # Resize
        ch = resize_image(ch, target_size)                  # [3, H', W']
        channels.append(ch)

    # Stack të gjitha slicet
    result = np.concatenate(channels, axis=0)               # [C, H, W]
    return result.astype(np.float32)


def preprocess_single_slice(
    hu_array:    np.ndarray,
    modality:    str = "CTA",
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Proceson një slice të vetëm → [3, 224, 224].
    Përdoret kur kemi slice specifike nga train_localizers.csv.

    Returns:
        np.ndarray [3, 224, 224] float32
    """
    windowed = apply_windowing(hu_array, modality)          # [H, W]
    three_ch = np.stack([windowed, windowed, windowed])     # [3, H, W]
    resized  = resize_image(three_ch, target_size)          # [3, 224, 224]
    return resized.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — NORMALIZIM ImageNet
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def normalize_imagenet(arr: np.ndarray) -> np.ndarray:
    """
    Aplikon ImageNet normalizim:
        output = (input - mean) / std

    Kjo është e nevojshme sepse ResNet-50/101 janë pretrained
    me ImageNet normalizim (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).

    Args:
        arr: [3, H, W] me vlera [0, 1]

    Returns:
        np.ndarray [3, H, W] normalized
    """
    out = arr.copy()
    for i in range(min(3, arr.shape[0])):
        out[i] = (arr[i] - IMAGENET_MEAN[i % 3]) / (IMAGENET_STD[i % 3] + 1e-8)
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — DATA AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

class DICOMAugmentor:
    """
    Augmentacion i lehtë i përshtatshëm për imazhe DICOM.

    Teknikat:
        - Horizontal Flip  (p=0.50) — neuroimaging simetrike
        - Rotation ±15°    (p=0.40) — variacion i pozicionimit
        - Brightness ±10%  (p=0.30) — variacion i skanimit
        - Gaussian Noise   (p=0.20) — robustësi ndaj noise
        - Zoom 0.9–1.1     (p=0.25) — variacion i madhësisë

    ⚠ VETËM për train set — kurrë për val/test!
    """

    def __init__(
        self,
        flip_prob:       float = 0.50,
        rotate_prob:     float = 0.40,
        rotate_range:    float = 15.0,
        brightness_prob: float = 0.30,
        brightness_range:float = 0.10,
        noise_prob:      float = 0.20,
        noise_std:       float = 0.02,
        zoom_prob:       float = 0.25,
        zoom_range:      Tuple = (0.90, 1.10),
    ):
        self.flip_prob        = flip_prob
        self.rotate_prob      = rotate_prob
        self.rotate_range     = rotate_range
        self.brightness_prob  = brightness_prob
        self.brightness_range = brightness_range
        self.noise_prob       = noise_prob
        self.noise_std        = noise_std
        self.zoom_prob        = zoom_prob
        self.zoom_range       = zoom_range

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        """
        Aplikon augmentacion random në array [C, H, W].

        Returns:
            np.ndarray [C, H, W] — i augmentuar
        """
        arr = arr.copy()

        # Horizontal Flip
        if np.random.random() < self.flip_prob:
            arr = arr[:, :, ::-1].copy()

        # Rotation
        if np.random.random() < self.rotate_prob:
            arr = self._rotate(arr)

        # Brightness
        if np.random.random() < self.brightness_prob:
            delta = np.random.uniform(
                -self.brightness_range,
                +self.brightness_range
            )
            arr = np.clip(arr + delta, 0.0, 1.0)

        # Gaussian Noise
        if np.random.random() < self.noise_prob:
            noise = np.random.normal(0, self.noise_std, arr.shape).astype(np.float32)
            arr   = np.clip(arr + noise, 0.0, 1.0)

        # Zoom
        if np.random.random() < self.zoom_prob:
            arr = self._zoom(arr)

        return arr.astype(np.float32)

    def _rotate(self, arr: np.ndarray) -> np.ndarray:
        """Rroton çdo kanal me të njëjtin kënd."""
        try:
            from scipy.ndimage import rotate as nd_rotate
            angle = np.random.uniform(-self.rotate_range, self.rotate_range)
            rotated = np.stack([
                nd_rotate(arr[i], angle, reshape=False, mode="nearest")
                for i in range(arr.shape[0])
            ])
            return rotated.astype(np.float32)
        except ImportError:
            return arr

    def _zoom(self, arr: np.ndarray) -> np.ndarray:
        """Zoom-on imazhin duke prerë ose duke mbushur me zero."""
        try:
            from scipy.ndimage import zoom as nd_zoom
            factor = np.random.uniform(*self.zoom_range)
            c, h, w = arr.shape
            zoomed = np.stack([
                nd_zoom(arr[i], factor, mode="nearest")
                for i in range(c)
            ])
            # Crop ose pad deri tek dimensioni origjinal
            zh, zw = zoomed.shape[1], zoomed.shape[2]
            out    = np.zeros((c, h, w), dtype=np.float32)
            ch_    = min(zh, h)
            cw_    = min(zw, w)
            out[:, :ch_, :cw_] = zoomed[:, :ch_, :cw_]
            return out
        except ImportError:
            return arr


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — BATCH PREPROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

class BatchPreprocessor:
    """
    Proceson batch-e të serive DICOM paralelisht.

    Integrohet me DataLoader:
        dataset = AneurysmDataset(df, series_dir, preprocessor=BatchPreprocessor())

    Args:
        target_size      : (H, W) input i modelit
        n_slices         : Slice per seri
        use_multi_window : True = multi-window 3 kanale per slice
        augmentor        : DICOMAugmentor (vetëm train)
        normalize        : True = apliko ImageNet normalizim
    """

    def __init__(
        self,
        target_size:      Tuple = (224, 224),
        n_slices:         int   = 3,
        use_multi_window: bool  = True,
        augmentor:        Optional[DICOMAugmentor] = None,
        normalize:        bool  = True,
    ):
        self.target_size      = target_size
        self.n_slices         = n_slices
        self.use_multi_window = use_multi_window
        self.augmentor        = augmentor
        self.normalize        = normalize

    def process_one(
        self,
        series_path: str,
        modality:    str = "CTA",
    ) -> np.ndarray:
        """
        Proceson një seri të vetme.

        Returns:
            np.ndarray [C, H, W] gati për model input
        """
        arr = preprocess_series(
            series_path      = series_path,
            modality         = modality,
            target_size      = self.target_size,
            n_slices         = self.n_slices,
            use_multi_window = self.use_multi_window,
        )

        if self.augmentor is not None:
            arr = self.augmentor(arr)

        # Merr vetëm 3 kanalet e para për ResNet-50/101
        if arr.shape[0] > 3:
            arr = arr[:3]

        if self.normalize:
            arr = normalize_imagenet(arr)

        return arr

    def process_batch(
        self,
        series_paths: List[str],
        modalities:   List[str],
    ) -> np.ndarray:
        """
        Proceson listë serishë → batch [B, 3, 224, 224].

        Args:
            series_paths : Lista e shtegëve
            modalities   : Lista e modaliteteve

        Returns:
            np.ndarray [B, 3, H, W]
        """
        batch = []
        for path, mod in zip(series_paths, modalities):
            arr = self.process_one(path, mod)
            batch.append(arr)
        return np.stack(batch, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# STATISTIKA TË PREPROCESSING-UT
# ─────────────────────────────────────────────────────────────────────────────

def compute_dataset_stats(
    df:         "pd.DataFrame",
    series_dir: str,
    n_samples:  int = 100,
) -> Dict:
    """
    Llogarit statistikat e pixel-ave mbi n_samples raste të rastit.
    Përdoret për të verifikuar normalizimin dhe window-settings.

    Returns:
        dict me mean, std, min, max per modalitet
    """
    stats = defaultdict(lambda: {"values": []})
    sample_df = df.sample(min(n_samples, len(df)), random_state=42)

    for _, row in sample_df.iterrows():
        sid      = row["SeriesInstanceUID"]
        mod      = row.get("Modality", "CTA")
        spath    = os.path.join(series_dir, sid)

        if not os.path.exists(spath):
            continue

        slices = read_dicom_series(spath)
        if slices:
            mid_slice = slices[len(slices) // 2]
            windowed  = apply_windowing(mid_slice["hu_array"], mod)
            stats[mod]["values"].extend(windowed.flatten()[:1000].tolist())

    result = {}
    for mod, data in stats.items():
        vals = np.array(data["values"])
        if len(vals) > 0:
            result[mod] = {
                "mean":    round(float(vals.mean()), 4),
                "std":     round(float(vals.std()),  4),
                "min":     round(float(vals.min()),  4),
                "max":     round(float(vals.max()),  4),
                "n_pixels": len(vals),
            }
            print(f"  {mod:<10}: mean={result[mod]['mean']:.4f}  "
                  f"std={result[mod]['std']:.4f}  "
                  f"[{result[mod]['min']:.2f}, {result[mod]['max']:.2f}]")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Demo dhe verifikim
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "█" * 65)
    print("  PREPROCESSING PIPELINE — NEUROVISION AI DETECTION")
    print("  Blina Sopjani | ID: 69401 | Universum College")
    print("█" * 65)

    print("\n" + "=" * 65)
    print("WINDOW SETTINGS SIPAS MODALITETIT")
    print("=" * 65)
    print(f"\n  {'Modaliteti':<12} {'WW (Width)':>12} {'WL (Level)':>12}  Arsyeja Klinike")
    print(f"  {'-'*60}")
    descriptions = {
        "CTA":    "Veçon enet e gjakut dhe ndryshimet vaskulare",
        "MRA":    "Flow-sensitive — MR Angiography signal",
        "MRI_T1": "Soft tissue contrast, post-contrast enhancement",
        "MRI_T2": "CSF dhe fluid visibility",
    }
    for mod, s in WINDOW_PRESETS.items():
        if mod == "DEFAULT": continue
        print(f"  {mod:<12} {s['ww']:>12} {s['wl']:>12}  {descriptions.get(mod,'')}")

    print("\n" + "=" * 65)
    print("DEMO — Synthetic Image Processing (pa DICOM real)")
    print("=" * 65)

    # Simulim i HU array-t
    hu_sim  = np.random.normal(50, 200, (512, 512)).astype(np.float32)
    hu_sim  = np.clip(hu_sim, -1000, 3000)

    print(f"\n  Input HU array  : {hu_sim.shape}  [{hu_sim.min():.0f}, {hu_sim.max():.0f}]")

    for mod in ["CTA", "MRA", "MRI_T1"]:
        windowed = apply_windowing(hu_sim, mod)
        print(f"  After windowing [{mod:<8}]: [{windowed.min():.3f}, {windowed.max():.3f}]  mean={windowed.mean():.4f}")

    # Multi-window
    multi = apply_multi_window(hu_sim, "CTA")
    print(f"\n  Multi-window CTA (3ch): {multi.shape}")

    # Resize
    resized = resize_image(multi, (224, 224))
    print(f"  After resize 224×224  : {resized.shape}")

    # Normalizim
    normed = normalize_imagenet(resized)
    print(f"  After ImageNet norm   : mean={normed.mean():.4f}  std={normed.std():.4f}")

    # Augmentation demo
    aug = DICOMAugmentor()
    augmented = aug(resized)
    print(f"\n  Augmentation output   : {augmented.shape}")

    print(f"""
  ── PIPELINE E PLOTË ─────────────────────────────────────
  from preprocessing import BatchPreprocessor, DICOMAugmentor

  # Train preprocessor (me augmentation)
  train_prep = BatchPreprocessor(
      target_size=(224, 224), n_slices=3,
      augmentor=DICOMAugmentor(), normalize=True
  )

  # Val/Test preprocessor (pa augmentation)
  val_prep = BatchPreprocessor(
      target_size=(224, 224), n_slices=3,
      augmentor=None, normalize=True
  )

  # Proceso një seri
  arr = train_prep.process_one('./series/1.2.826.../', modality='CTA')
  # arr.shape = (3, 224, 224) gati për model input
  ──────────────────────────────────────────────────────────
    """)
