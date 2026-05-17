"""
=============================================================================
RSNA 2025 — DATASET CONNECTION & LOADING MODULE
=============================================================================
Student   : Blina Sopjani | ID: 69401
Dataset   : rsna-intracranial-aneurysm-detection (Kaggle Competition)

LIDHJA ME DATASET — 3 METODA:
  Metoda A — kagglehub  (rekomandohet, automatike)
  Metoda B — Kaggle CLI (manual download)
  Metoda C — Kaggle MCP Server

STRUKTURA E DATASET-IT (nga Dataset_Description.txt):
  train.csv              — labels kryesore (SeriesInstanceUID + 14 kolona)
  train_localizers.csv   — koordinatat e aneurizmave (SOPInstanceUID + xy)
  series/                — DICOM files: series/{SeriesInstanceUID}/{SOPInstanceUID}.dcm
  segmentations/         — NifTI vessel segmentations (subset)

DICOM TAGS (të disponueshme në test set):
  BitsAllocated, BitsStored, Columns, FrameOfReferenceUID, HighBit,
  ImageOrientationPatient, ImagePositionPatient, InstanceNumber, Modality,
  PatientID, PixelRepresentation, PixelSpacing, RescaleIntercept,
  RescaleSlope, Rows, SOPInstanceUID, SliceThickness, StudyInstanceUID,
  TransferSyntaxUID, ...

REQUIREMENTS:
  pip install kagglehub kaggle pandas numpy pydicom nibabel tqdm
=============================================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURIM
# ─────────────────────────────────────────────────────────────────────────────

COMPETITION_ID = "rsna-intracranial-aneurysm-detection"

# Kolonat target (nga Dataset_Description.txt)
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

# DICOM tags të lejuara (nga dokumentacioni zyrtar)
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

# Segmentation label map (nga Dataset_Description.txt)
SEGMENTATION_LABELS = {
    1:  "Other Posterior Circulation",
    2:  "Basilar Tip",
    3:  "Right Posterior Communicating Artery",
    4:  "Left Posterior Communicating Artery",
    5:  "Right Infraclinoid Internal Carotid Artery",
    6:  "Left Infraclinoid Internal Carotid Artery",
    7:  "Right Supraclinoid Internal Carotid Artery",
    8:  "Left Supraclinoid Internal Carotid Artery",
    9:  "Right Middle Cerebral Artery",
    10: "Left Middle Cerebral Artery",
    11: "Right Anterior Cerebral Artery",
    12: "Left Anterior Cerebral Artery",
    13: "Anterior Communicating Artery",
}


# ─────────────────────────────────────────────────────────────────────────────
# METODA A — kagglehub (REKOMANDOHET)
# ─────────────────────────────────────────────────────────────────────────────

def connect_via_kagglehub() -> str:
    """
    Shkarkon dataset-in automatikisht me kagglehub.

    Kërkesat:
        pip install kagglehub
        Konfiguro Kaggle API token: ~/.kaggle/kaggle.json
            {
              "username": "your_username",
              "key": "your_api_key"
            }
        Merr API key nga: https://www.kaggle.com/settings -> API

    Returns:
        str: Shtegu lokal i dataset-it
    """
    print("=" * 65)
    print("METODA A — Lidhja me kagglehub")
    print("=" * 65)

    try:
        import kagglehub  # pip install kagglehub

        print("  → Shkarkim i dataset-it...")
        path = kagglehub.competition_download(COMPETITION_ID)
        print(f"  ✅ Dataset i shkarkuar në: {path}")
        return path

    except ImportError:
        print("  ❌ kagglehub nuk është instaluar.")
        print("     Zgjidhe: pip install kagglehub")
        return None
    except Exception as e:
        print(f"  ❌ Gabim: {e}")
        print("     Kontrollo ~/.kaggle/kaggle.json")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# METODA B — Kaggle CLI (manual)
# ─────────────────────────────────────────────────────────────────────────────

def connect_via_kaggle_cli(output_dir: str = "./data/") -> str:
    """
    Shkarkon dataset-in me Kaggle CLI.

    Kërkesat:
        pip install kaggle
        Vendos kaggle.json në ~/.kaggle/kaggle.json
        chmod 600 ~/.kaggle/kaggle.json

    Komanda ekuivalente bash:
        kaggle competitions download -c rsna-intracranial-aneurysm-detection
        unzip rsna-intracranial-aneurysm-detection.zip -d ./data/

    Returns:
        str: Shtegu lokal pasi shkarkohet
    """
    print("=" * 65)
    print("METODA B — Lidhja me Kaggle CLI")
    print("=" * 65)

    try:
        import subprocess

        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            "kaggle", "competitions", "download",
            "-c", COMPETITION_ID,
            "-p", output_dir
        ]
        print(f"  → Ekzekuton: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  ✅ Shkarkuar në: {output_dir}")
            # Shpaketim automatik
            zip_file = os.path.join(output_dir, f"{COMPETITION_ID}.zip")
            if os.path.exists(zip_file):
                import zipfile
                print(f"  → Shpaketim: {zip_file}")
                with zipfile.ZipFile(zip_file, "r") as z:
                    z.extractall(output_dir)
                print(f"  ✅ Shpaketuar në: {output_dir}")
            return output_dir
        else:
            print(f"  ❌ CLI Error: {result.stderr}")
            print("     Kontrollo: kaggle --version dhe kaggle.json")
            return None

    except FileNotFoundError:
        print("  ❌ Kaggle CLI nuk është instaluar.")
        print("     Zgjidhe: pip install kaggle")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# METODA C — Kaggle MCP Server
# ─────────────────────────────────────────────────────────────────────────────

def connect_via_mcp_instructions():
    """
    Udhëzime për lidhjen me Kaggle MCP Server.

    MCP (Model Context Protocol) Server URL:
        https://www.kaggle.com/mcp

    Tool të përdorësh:
        mcp_kaggle_download_competition_data_files

    Hapat:
        1. Hap klientin tënd MCP (Claude, VS Code, etc.)
        2. Lidhu me serverin: https://www.kaggle.com/mcp
        3. Thirr toolin: mcp_kaggle_download_competition_data_files
        4. Argumento competition_id: rsna-intracranial-aneurysm-detection
    """
    print("=" * 65)
    print("METODA C — Kaggle MCP Server")
    print("=" * 65)
    print("""
  URL MCP Server : https://www.kaggle.com/mcp
  Tool           : mcp_kaggle_download_competition_data_files
  Competition ID : rsna-intracranial-aneurysm-detection

  Hapat në Claude / VS Code:
    1. Shko te Settings → MCP Servers
    2. Shto: https://www.kaggle.com/mcp
    3. Thirr: mcp_kaggle_download_competition_data_files
    4. Vendos competition_id = rsna-intracranial-aneurysm-detection
    """)


# ─────────────────────────────────────────────────────────────────────────────
# VERIFIKIMI I DATASET-IT
# ─────────────────────────────────────────────────────────────────────────────

def verify_dataset_structure(base_path: str) -> dict:
    """
    Verifikon strukturën e dataset-it pas shkarkimit.

    Struktura e pritur (nga Dataset_Description.txt):
        {base_path}/
        ├── train.csv
        ├── train_localizers.csv
        ├── series/
        │   └── {SeriesInstanceUID}/
        │       └── {SOPInstanceUID}.dcm
        └── segmentations/
            └── {SeriesInstanceUID}.nii.gz

    Args:
        base_path: Shtegu bazë i dataset-it

    Returns:
        dict: Raport verifikimi me statistika
    """
    print("\n" + "=" * 65)
    print("VERIFIKIMI I STRUKTURËS SË DATASET-IT")
    print("=" * 65)

    report = {"base_path": base_path, "status": "OK", "issues": []}
    base = Path(base_path)

    # Kontrollo skedarët kryesorë
    required_files = {
        "train.csv":            "Labels kryesore + demografika",
        "train_localizers.csv": "Koordinatat e aneurizmave",
    }

    print(f"\n  📁 Shtegu bazë: {base_path}")
    print(f"\n  Skedarë kryesorë:")
    for fname, desc in required_files.items():
        fpath = base / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1024 / 1024
            print(f"    ✅ {fname:<30} ({size_mb:.2f} MB) — {desc}")
            report[fname] = {"exists": True, "size_mb": round(size_mb, 2)}
        else:
            print(f"    ❌ {fname:<30} — MUNGON")
            report["issues"].append(f"Mungon: {fname}")
            report[fname] = {"exists": False}

    # Kontrollo direktorinë series/
    series_dir = base / "series"
    if series_dir.exists():
        series_ids = [d.name for d in series_dir.iterdir() if d.is_dir()]
        dcm_total  = sum(len(list(d.glob("*.dcm"))) for d in series_dir.iterdir() if d.is_dir())
        print(f"\n  📂 series/ direktorium:")
        print(f"    ✅ Serje të gjetura   : {len(series_ids):,}")
        print(f"    ✅ Skedarë .dcm total : {dcm_total:,}")
        print(f"    ✅ Shembull SeriesUID  : {series_ids[0] if series_ids else 'N/A'}")
        report["series"] = {"n_series": len(series_ids), "n_dcm": dcm_total}
    else:
        print(f"\n    ⚠ series/ direktorium nuk u gjet (mund të jetë Kaggle-only)")
        report["series"] = {"exists": False}

    # Kontrollo segmentations/
    seg_dir = base / "segmentations"
    if seg_dir.exists():
        nii_files = list(seg_dir.glob("*.nii.gz")) + list(seg_dir.glob("*.nii"))
        print(f"\n  📂 segmentations/ direktorium:")
        print(f"    ✅ Skedarë NifTI : {len(nii_files):,}")
        report["segmentations"] = {"n_files": len(nii_files)}
    else:
        print(f"\n    ℹ segmentations/ — nuk është disponueshëm (subset opsional)")
        report["segmentations"] = {"exists": False}

    if not report["issues"]:
        print(f"\n  ✅ Struktura e dataset-it VERIFIKUAR me sukses!")
    else:
        print(f"\n  ⚠ {len(report['issues'])} probleme të gjetura: {report['issues']}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# LEXIMI I train.csv
# ─────────────────────────────────────────────────────────────────────────────

def load_train_csv(base_path: str) -> pd.DataFrame:
    """
    Lexon dhe validizon train.csv.

    Struktura e pritur (nga Dataset_Description.txt):
        SeriesInstanceUID    — ID unike e serisë
        Modality             — CTA / MRA / MRI
        PatientAge           — Mosha e pacientit
        PatientSex           — Gjinia (M/F)
        [13 location cols]   — Binary (0/1) për çdo lokacion
        Aneurysm Present     — Target kryesor (0/1)

    Returns:
        pd.DataFrame: Dataset i ngarkuar dhe i validuar
    """
    print("\n" + "=" * 65)
    print("LEXIMI I train.csv")
    print("=" * 65)

    train_path = os.path.join(base_path, "train.csv")

    if not os.path.exists(train_path):
        print(f"  ❌ train.csv nuk u gjet në: {train_path}")
        print("     → Duke gjeneruar dataset simulues për analizë...")
        return _simulate_train_csv()

    df = pd.read_csv(train_path)
    print(f"\n  ✅ train.csv i ngarkuar: {df.shape[0]:,} rreshta × {df.shape[1]} kolona")

    # Kontrollo kolonat e pritura
    print(f"\n  Kolonat e pritura vs. të gjetura:")
    expected = ["SeriesInstanceUID", "Modality", "PatientAge", "PatientSex"] + LABEL_COLS
    missing_cols = [c for c in expected if c not in df.columns]
    extra_cols   = [c for c in df.columns if c not in expected]

    if not missing_cols:
        print(f"    ✅ Të gjitha {len(expected)} kolonat janë prezente")
    else:
        print(f"    ❌ Kolona mungese: {missing_cols}")

    if extra_cols:
        print(f"    ℹ  Kolona shtesë: {extra_cols}")

    # Statistika bazë
    print(f"\n  Statistika bazë:")
    print(f"    Total serje         : {len(df):,}")
    print(f"    Aneurysm Positive   : {df['Aneurysm Present'].sum():,} ({df['Aneurysm Present'].mean()*100:.1f}%)")
    print(f"    Aneurysm Negative   : {(df['Aneurysm Present']==0).sum():,} ({(1-df['Aneurysm Present'].mean())*100:.1f}%)")
    print(f"    Modalitete          : {df['Modality'].unique().tolist()}")
    print(f"    Missing (PatientAge): {df['PatientAge'].isna().sum()} ({df['PatientAge'].isna().mean()*100:.1f}%)")
    print(f"    Missing (PatientSex): {df['PatientSex'].isna().sum()} ({df['PatientSex'].isna().mean()*100:.1f}%)")

    return df


def _simulate_train_csv() -> pd.DataFrame:
    """
    Gjeneron train.csv simulues me strukturë identike me dataset-in real.
    Përdoret kur dataset-i nuk është shkarkuar ende.
    """
    import numpy as np
    np.random.seed(42)
    N = 3200

    aneurysm = np.random.choice([0, 1], N, p=[0.786, 0.214])
    mod       = np.random.choice(["CTA", "MRA", "MRI_T1", "MRI_T2"], N, p=[0.45, 0.30, 0.15, 0.10])
    age       = np.random.normal(54, 15, N).clip(18, 90).astype(float)
    sex       = np.random.choice(["F", "M"], N, p=[0.56, 0.44]).astype(object)

    age[np.random.choice(N, int(N*0.043), replace=False)] = np.nan
    sex[np.random.choice(N, int(N*0.021), replace=False)] = None

    loc_data = {
        c: np.where(aneurysm==1, np.random.choice([0,1], N, p=[0.80,0.20]), 0)
        for c in LABEL_COLS[:-1]
    }

    df = pd.DataFrame({
        "SeriesInstanceUID": [f"1.2.826.0.1.{i:012d}" for i in range(N)],
        "Modality":          mod,
        "PatientAge":        age,
        "PatientSex":        sex,
        **loc_data,
        "Aneurysm Present":  aneurysm,
    })

    print(f"  ✅ Dataset simulues: {df.shape[0]:,} rreshta × {df.shape[1]} kolona")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LEXIMI I train_localizers.csv
# ─────────────────────────────────────────────────────────────────────────────

def load_localizers_csv(base_path: str) -> pd.DataFrame:
    """
    Lexon train_localizers.csv — koordinatat e aneurizmave.

    Struktura (nga Dataset_Description.txt):
        SeriesInstanceUID  — ID e serisë (FK -> train.csv)
        SOPInstanceUID     — ID e imazhit specifik brenda serisë
        coordinates        — xy koordinatat e qendrës së aneurizmës
        location           — Emri tekst i lokacionit anatomik

    Returns:
        pd.DataFrame: Lokalizer data
    """
    print("\n" + "=" * 65)
    print("LEXIMI I train_localizers.csv")
    print("=" * 65)

    loc_path = os.path.join(base_path, "train_localizers.csv")

    if not os.path.exists(loc_path):
        print(f"  ⚠ train_localizers.csv nuk u gjet — dataset jo i shkarkuar")
        return pd.DataFrame()

    df_loc = pd.read_csv(loc_path)
    print(f"  ✅ train_localizers.csv: {df_loc.shape[0]:,} aneurizma të lokalizuara")
    print(f"  Kolonat : {df_loc.columns.tolist()}")
    print(f"  Shembull :\n{df_loc.head(3).to_string(index=False)}")

    return df_loc


# ─────────────────────────────────────────────────────────────────────────────
# LEXIMI I DICOM SKEDARËVE
# ─────────────────────────────────────────────────────────────────────────────

def load_dicom_series(series_path: str) -> dict:
    """
    Lexon të gjitha skedarët DICOM nga një folder serje.

    Shtegu: series/{SeriesInstanceUID}/{SOPInstanceUID}.dcm

    DICOM Tags të lexuara (nga DICOM_TAG_ALLOWLIST):
        - Modality, SliceThickness, PixelSpacing
        - RescaleSlope, RescaleIntercept (për HU conversion)
        - Rows, Columns (dimensionet e imazhit)
        - ImagePositionPatient (pozicioni 3D i çdo slice)
        - PixelData -> pixel_array (imazhi aktual)

    Args:
        series_path: Shtegu i folderit të serisë

    Returns:
        dict me metadata dhe pixel arrays të renditura sipas pozicionit
    """
    try:
        import pydicom
    except ImportError:
        print("  ❌ pydicom nuk është instaluar: pip install pydicom")
        return {}

    series_id = os.path.basename(series_path)
    dcm_files = sorted(Path(series_path).glob("*.dcm"))

    if not dcm_files:
        return {"series_id": series_id, "n_slices": 0, "slices": []}

    slices     = []
    metadata   = {}
    tags       = defaultdict(list)

    for dcm_path in dcm_files:
        ds = pydicom.dcmread(str(dcm_path), force=True)

        # Lexo tags nga allowlist
        for tag in DICOM_TAG_ALLOWLIST:
            val = getattr(ds, tag, None)
            if val is not None:
                tags[tag].append(val)

        # Lexo pixel array (imazhi)
        try:
            pixel_array = ds.pixel_array.astype(np.float32)

            # HU conversion
            slope     = float(getattr(ds, "RescaleSlope",     1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            hu_array  = pixel_array * slope + intercept

            # Pozicioni i slice-it (për renditje)
            pos = getattr(ds, "ImagePositionPatient", [0, 0, dcm_path.stem])
            z   = float(pos[2]) if hasattr(pos, "__len__") else 0.0

            slices.append({
                "sop_uid":  getattr(ds, "SOPInstanceUID", dcm_path.stem),
                "filepath": str(dcm_path),
                "z_pos":    z,
                "hu_array": hu_array,
            })
        except Exception:
            pass

    # Rendit slices sipas pozicionit Z (kraniokaudal)
    slices.sort(key=lambda s: s["z_pos"])

    # Metadata kryesore nga slice-i i parë
    if dcm_files:
        ds0 = pydicom.dcmread(str(dcm_files[0]), force=True)
        metadata = {
            "series_id":       series_id,
            "modality":        getattr(ds0, "Modality",        "UNKNOWN"),
            "rows":            getattr(ds0, "Rows",            None),
            "columns":         getattr(ds0, "Columns",         None),
            "pixel_spacing":   getattr(ds0, "PixelSpacing",    None),
            "slice_thickness": getattr(ds0, "SliceThickness",  None),
            "n_slices":        len(slices),
        }

    return {**metadata, "slices": slices, "tags": dict(tags)}


def load_all_series(base_path: str, df_train: pd.DataFrame,
                    max_series: int = None) -> dict:
    """
    Lexon të gjitha seritë DICOM të listuara në train.csv.

    Args:
        base_path  : Shtegu bazë i dataset-it
        df_train   : DataFrame i ngarkuar nga train.csv
        max_series : Numri maksimal i serive për t'u lexuar (None = të gjitha)

    Returns:
        dict: {series_id -> dicom_data}
    """
    series_dir = os.path.join(base_path, "series")
    if not os.path.exists(series_dir):
        print(f"  ⚠ series/ direktorium nuk u gjet në: {series_dir}")
        return {}

    series_ids = df_train["SeriesInstanceUID"].tolist()
    if max_series:
        series_ids = series_ids[:max_series]

    try:
        from tqdm import tqdm
        iterator = tqdm(series_ids, desc="  Lexim DICOM", unit="serje")
    except ImportError:
        iterator = series_ids

    all_data = {}
    for sid in iterator:
        spath = os.path.join(series_dir, sid)
        if os.path.exists(spath):
            all_data[sid] = load_dicom_series(spath)

    print(f"\n  ✅ Serje DICOM të lexuara: {len(all_data):,}")
    return all_data


# ─────────────────────────────────────────────────────────────────────────────
# LEXIMI I SEGMENTIMEVE NifTI
# ─────────────────────────────────────────────────────────────────────────────

def load_segmentation(series_id: str, base_path: str) -> dict:
    """
    Lexon segmentimin NifTI për një seri (nëse ekziston).

    Shtegu: segmentations/{SeriesInstanceUID}.nii.gz

    Label Map (nga Dataset_Description.txt):
        1  = Other Posterior Circulation
        2  = Basilar Tip
        3  = Right Posterior Communicating Artery
        4  = Left Posterior Communicating Artery
        5  = Right Infraclinoid Internal Carotid Artery
        6  = Left Infraclinoid Internal Carotid Artery
        7  = Right Supraclinoid Internal Carotid Artery
        8  = Left Supraclinoid Internal Carotid Artery
        9  = Right Middle Cerebral Artery
        10 = Left Middle Cerebral Artery
        11 = Right Anterior Cerebral Artery
        12 = Left Anterior Cerebral Artery
        13 = Anterior Communicating Artery

    Returns:
        dict me volumin 3D dhe label-at prezente
    """
    seg_path = os.path.join(base_path, "segmentations", f"{series_id}.nii.gz")
    if not os.path.exists(seg_path):
        seg_path = seg_path.replace(".nii.gz", ".nii")

    if not os.path.exists(seg_path):
        return {"series_id": series_id, "has_segmentation": False}

    try:
        import nibabel as nib  # pip install nibabel
        img  = nib.load(seg_path)
        data = img.get_fdata().astype(np.int32)

        # Gji label-at prezente
        unique_labels = np.unique(data)
        unique_labels  = unique_labels[unique_labels > 0]
        label_names    = [SEGMENTATION_LABELS.get(int(l), f"Unknown_{l}") for l in unique_labels]

        return {
            "series_id":        series_id,
            "has_segmentation": True,
            "seg_path":         seg_path,
            "volume_shape":     data.shape,
            "voxel_size_mm":    img.header.get_zooms()[:3],
            "labels_present":   {int(l): SEGMENTATION_LABELS.get(int(l)) for l in unique_labels},
            "label_names":      label_names,
            "volume":           data,
        }

    except ImportError:
        print("  ❌ nibabel nuk është instaluar: pip install nibabel")
        return {"series_id": series_id, "has_segmentation": False}
    except Exception as e:
        print(f"  ❌ Gabim në lexim segmentimi [{series_id}]: {e}")
        return {"series_id": series_id, "has_segmentation": False}


# ─────────────────────────────────────────────────────────────────────────────
# DATASET STATISTICS REPORT
# ─────────────────────────────────────────────────────────────────────────────

def dataset_statistics(df: pd.DataFrame) -> dict:
    """
    Gjeneron statistika të plota të dataset-it pas lidhjes.

    Args:
        df: DataFrame i ngarkuar nga train.csv

    Returns:
        dict me të gjitha statistikat
    """
    print("\n" + "=" * 65)
    print("STATISTIKA TË DATASET-IT PAS LIDHJES")
    print("=" * 65)

    N   = len(df)
    pos = int(df["Aneurysm Present"].sum())
    neg = N - pos

    stats = {
        "total_series": N,
        "positive":     pos,
        "negative":     neg,
        "prevalence":   round(pos / N * 100, 2),
        "imbalance":    round(neg / pos, 2),
    }

    print(f"\n  📊 Shpërndarja e Klasave:")
    print(f"     Total serje      : {N:,}")
    print(f"     Pozitive         : {pos:,}  ({stats['prevalence']}%)")
    print(f"     Negative         : {neg:,}  ({100-stats['prevalence']}%)")
    print(f"     Raporti imbalancë: 1:{stats['imbalance']}")

    print(f"\n  🔬 Modalitetet:")
    mod_stats = df.groupby("Modality")["Aneurysm Present"].agg(
        Total="count",
        Positive="sum"
    )
    mod_stats["Prevalence%"] = (mod_stats["Positive"] / mod_stats["Total"] * 100).round(1)
    for mod, row in mod_stats.iterrows():
        print(f"     {mod:<10}: {int(row['Total']):5,} serje | {int(row['Positive']):4,} pozitive ({row['Prevalence%']}%)")
    stats["modality"] = mod_stats.to_dict()

    print(f"\n  📍 Lokacionet (frekuenca):")
    loc_cols  = LABEL_COLS[:-1]
    loc_freq  = df[loc_cols].sum().sort_values(ascending=False)
    for loc, cnt in loc_freq.head(5).items():
        print(f"     {loc[:38]:38s}: {int(cnt):4d} ({cnt/pos*100:.1f}%)")
    stats["location_freq"] = loc_freq.to_dict()

    print(f"\n  👤 Demografika:")
    print(f"     Mosha mesatare   : {df['PatientAge'].mean():.1f} ± {df['PatientAge'].std():.1f} vjet")
    print(f"     Female           : {(df['PatientSex']=='F').sum():,} ({(df['PatientSex']=='F').mean()*100:.1f}%)")
    print(f"     Male             : {(df['PatientSex']=='M').sum():,} ({(df['PatientSex']=='M').mean()*100:.1f}%)")

    print(f"\n  ❗ Missing Values:")
    mv = df.isnull().sum()
    for col, n in mv[mv > 0].items():
        print(f"     {col:<25}: {n} ({n/N*100:.1f}%)")
    if mv.sum() == 0:
        print(f"     Asnjë missing value!")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# FUNKSIONI KRYESOR — auto-detect metoda
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(method: str = "auto", custom_path: str = None) -> dict:
    """
    Funksioni kryesor — lidhet me dataset-in duke provuar metodat sipas radhës.

    Args:
        method      : "auto" | "kagglehub" | "cli" | "mcp" | "local"
        custom_path : Shtegu manual nëse dataset-i është shkarkuar tashmë

    Returns:
        dict me:
            - df_train     : pd.DataFrame i train.csv
            - df_localizers: pd.DataFrame i train_localizers.csv
            - base_path    : shtegu lokal i dataset-it
            - stats        : statistikat e dataset-it
    """
    print("\n" + "=" * 65)
    print("  RSNA DATASET - LIDHJA DHE NGARKIMI I TE DHENAVE")
    print("  Blina Sopjani | ID: 69401 | Universum College")
    print("=" * 65)

    base_path = None

    # 1. Nëse ka shteg manual
    if custom_path and os.path.exists(custom_path):
        base_path = custom_path
        print(f"\n  ✅ Duke përdorur shteg lokal: {base_path}")

    # 2. Kërko në vendndodhje standarde
    if not base_path:
        candidates = [
            f"./data/{COMPETITION_ID}/",
            f"./data/",
            f"~/kaggle/{COMPETITION_ID}/",
            f"/kaggle/input/{COMPETITION_ID}/",   # Kaggle Notebook environment
        ]
        for c in candidates:
            expanded = os.path.expanduser(c)
            if os.path.exists(expanded) and os.path.exists(os.path.join(expanded, "train.csv")):
                base_path = expanded
                print(f"\n  ✅ Dataset gjetur lokalisht: {base_path}")
                break

    # 3. Shkarko nëse jo i gjetur
    if not base_path:
        if method in ("auto", "kagglehub"):
            base_path = connect_via_kagglehub()
        if not base_path and method in ("auto", "cli"):
            base_path = connect_via_kaggle_cli()
        if not base_path:
            connect_via_mcp_instructions()
            print("\n  ⚠ Duke simuluar dataset-in për zhvillim lokal...")
            base_path = "./data_simulated/"
            os.makedirs(base_path, exist_ok=True)

    # 4. Verifiko strukturën
    if os.path.exists(base_path):
        verify_dataset_structure(base_path)

    # 5. Ngarko të dhënat
    df_train      = load_train_csv(base_path)
    df_localizers = load_localizers_csv(base_path)
    stats         = dataset_statistics(df_train)

    print("\n" + "=" * 65)
    print("  OK LIDHJA ME DATASET PERFUNDOI ME SUKSES")
    print(f"  {stats['total_series']:,} serje | {stats['positive']:,} pozitive | {stats['prevalence']}% prevalence")
    print("=" * 65 + "\n")

    return {
        "df_train":      df_train,
        "df_localizers": df_localizers,
        "base_path":     base_path,
        "stats":         stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Lidhja me dataset (metoda automatike) ─────────────────────────────
    data = load_dataset(method="auto")

    df_train = data["df_train"]
    base     = data["base_path"]
    stats    = data["stats"]

    # ── Shembull: lexo DICOM për 1 seri ──────────────────────────────────
    print("=" * 65)
    print("SHEMBULL: Leximi i 1 serje DICOM")
    print("=" * 65)

    sample_sid  = df_train["SeriesInstanceUID"].iloc[0]
    sample_path = os.path.join(base, "series", sample_sid)

    if os.path.exists(sample_path):
        dicom_data = load_dicom_series(sample_path)
        print(f"  SeriesID   : {dicom_data.get('series_id')}")
        print(f"  Modality   : {dicom_data.get('modality')}")
        print(f"  Slices     : {dicom_data.get('n_slices')}")
        print(f"  Dimensions : {dicom_data.get('rows')} × {dicom_data.get('columns')}")
        print(f"  PixelSpacing: {dicom_data.get('pixel_spacing')}")
    else:
        print(f"  ℹ DICOM skedarët janë disponueshëm vetëm pas shkarkimit të plotë.")
        print(f"  → Shkarko me: kaggle competitions download -c {COMPETITION_ID}")

    # ── Shembull: kontrollo segmentimin ───────────────────────────────────
    print("\n" + "=" * 65)
    print("SHEMBULL: Kontrollo segmentimin")
    print("=" * 65)

    seg_result = load_segmentation(sample_sid, base)
    if seg_result["has_segmentation"]:
        print(f"  ✅ Segmentim gjetur: {seg_result['label_names']}")
    else:
        print(f"  ℹ Segmentimi NifTI disponohet vetëm për subset të serive.")
        print(f"  → Disponueshëm në: segmentations/{{SeriesInstanceUID}}.nii.gz")

    # ── Eksporto statistikat ─────────────────────────────────────────────
    os.makedirs("./outputs/", exist_ok=True)
    exportable = {
        k: (v if not isinstance(v, pd.DataFrame) else "DataFrame")
        for k, v in stats.items()
        if not isinstance(v, dict) or k == "location_freq"
    }

    print(f"\n  ✅ Lidhja dhe ngarkimi i dataset-it — PËRFUNDOI")
    print(f"  → Tani vazhdo me: rsna_aneurysm_analysis.py")
