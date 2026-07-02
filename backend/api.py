
import os
import sys
import json
import warnings
import numpy as np
import io

warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# Shto scripts/ në path për import të modelit
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

# ─────────────────────────────────────────────────────────────────────────────
# TORCH IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None

# ─────────────────────────────────────────────────────────────────────────────
# KONSTANTET — 14 label outputs (saktë si RSNA)
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
    "Aneurysm Present",          # Target kryesor — indeksi 13
]

N_CLASSES       = len(LABEL_COLS)   # 14
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "CNNBaseline_best.pt"
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION — identike me model_training.py (ResNetModel)
# ─────────────────────────────────────────────────────────────────────────────

def build_resnet101(n_classes: int = 14) -> "nn.Module":
    """
    Ndërton arkitekturën ResNet-101 me classification head 14-output.
    Identike me ResNetModel nga model_training.py.
    """
    if not TORCH_AVAILABLE:
        return None

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
        nn.Linear(128, n_classes),    # 14 outputs
    )

    # Inicializo head me Xavier
    for m in head.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    class ResNet101Multilabel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.pool     = pool
            self.head     = head

        def forward(self, x):
            x = self.backbone(x)   # [B, 2048, 7, 7]
            x = self.pool(x)       # [B, 2048, 1, 1]
            x = self.head(x)       # [B, 14]
            return torch.sigmoid(x)  # Probabilitete [0,1] per çdo label

    return ResNet101Multilabel()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER — singleton
# ─────────────────────────────────────────────────────────────────────────────

_model_cache = None
_model_loaded_from_checkpoint = False

def get_model():
    """Ngarkon modelin njëherë dhe e mban në memorie (singleton pattern)."""
    global _model_cache, _model_loaded_from_checkpoint

    if _model_cache is not None:
        return _model_cache

    if not TORCH_AVAILABLE:
        print("  ⚠ PyTorch nuk disponueshëm — inference do kthejë vlera demo")
        return None

    model = build_resnet101(n_classes=N_CLASSES)

    ckpt_path = os.path.abspath(CHECKPOINT_PATH)
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state"])
            _model_loaded_from_checkpoint = True
            val_auc = ckpt.get('val_auc', None)
            auc_str = f"{val_auc:.4f}" if val_auc is not None else "?"
            print(f"  ✅ Checkpoint ngarkuar: epoch={ckpt.get('epoch','?')}, "
                  f"AUC={auc_str}")
        except Exception as e:
            print(f"  ⚠ Gabim checkpoint: {e} — duke përdorur peshë random")
    else:
        print(f"  ⚠ Checkpoint nuk u gjet: {ckpt_path}")
        print(f"     Trajno modelin me: python scripts/model_training.py")
        print(f"     API funksionon me probabilitete demo deri atëherë.")

    model.eval()
    model.to(DEVICE)
    _model_cache = model
    return model


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

preprocess_transform = None

def get_transform():
    """Kthen transform pipeline për inference (pa augmentation)."""
    global preprocess_transform
    if preprocess_transform is None and TORCH_AVAILABLE:
        preprocess_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return preprocess_transform


def preprocess_image(img_bytes: bytes) -> "torch.Tensor | None":
    """
    Konverton bytes të imazhit → tensor [1, 3, 224, 224].

    Pranon: PNG, JPEG, imazhe grayscale ose RGB.
    Grayscale replikohet në 3 kanale (RGB) për ResNet input.
    """
    if not TORCH_AVAILABLE:
        return None

    try:
        image = Image.open(io.BytesIO(img_bytes))

        # Konverto në RGB (ResNet pret 3 kanale)
        if image.mode == "L":             # grayscale → RGB
            image = image.convert("RGB")
        elif image.mode == "RGBA":        # RGBA → RGB
            image = image.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        transform = get_transform()
        tensor    = transform(image)      # [3, 224, 224]
        return tensor.unsqueeze(0)        # [1, 3, 224, 224]

    except Exception as e:
        print(f"  ⚠ Gabim preprocessing: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)   # Lejon komunikimin cross-origin me Dashboard-in

# Ngarko modelin kur starton serveri
print("\n  NeuroVision AI — Duke ngarkuar modelin...")
_inference_model = get_model()


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1: /health — Status check
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Kontrollo statusin e serverit dhe modelit."""
    return jsonify({
        "status":       "running",
        "model":        "CNN Baseline",
        "n_outputs":    N_CLASSES,
        "torch":        TORCH_AVAILABLE,
        "device":       str(DEVICE) if DEVICE else "N/A",
        "checkpoint":   _model_loaded_from_checkpoint,
        "labels":       LABEL_COLS,
    })


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2: /model_info — Info mbi modelin
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/model_info", methods=["GET"])
def model_info():
    """Kthen info të detajuar mbi modelin dhe output labels."""
    label_info = []
    for i, label in enumerate(LABEL_COLS):
        label_info.append({
            "index":  i,
            "label":  label,
            "weight": 13 if label == "Aneurysm Present" else 1,
            "type":   "primary" if label == "Aneurysm Present" else "location",
        })

    return jsonify({
        "model_name":    "CNN Baseline",
        "architecture":  "CNN Baseline architecture",
        "n_classes":     N_CLASSES,
        "input_size":    "224×224×3",
        "output":        "14 probabilitete sigmoid [0.0, 1.0]",
        "metric":        "RSNA Weighted Columnwise AUCROC",
        "labels":        label_info,
        "checkpoint":    _model_loaded_from_checkpoint,
        "device":        str(DEVICE) if DEVICE else "cpu",
    })


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3: /predict — Inference kryesor (imazh upload)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Inference mbi imazh të ngarkuar.

    Request:
        multipart/form-data me fushën 'file' (PNG/JPEG/imazh mjekësor)

    Response JSON:
        {
            "aneurysm_present": bool,
            "confidence": float (0-100),
            "probabilities": {label: prob, ...},  # të 14 label-at
            "detected_locations": [str, ...],      # lokacionet me prob > 0.5
            "model": "ResNet-101",
            "n_outputs": 14,
            "status": "success"
        }
    """
    if "file" not in request.files:
        return jsonify({"error": "Nuk u dërgua fajll. Shto 'file' në request."}), 400

    file      = request.files["file"]
    img_bytes = file.read()

    if not img_bytes:
        return jsonify({"error": "Fajlli është bosh."}), 400

    # Preproceso imazhin
    tensor = preprocess_image(img_bytes)

    model  = get_model()

    if model is not None and tensor is not None and TORCH_AVAILABLE:
        try:
            with torch.no_grad():
                tensor = tensor.to(DEVICE)
                output = model(tensor)                     # [1, 14]
                probs  = output.squeeze(0).cpu().numpy()   # [14]
        except Exception as e:
            print(f"  ⚠ Inference gabim: {e}")
            probs = np.full(N_CLASSES, 0.5, dtype=np.float32)
    else:
        # Demo mode — probabilitete sintetike reale
        np.random.seed(int.from_bytes(img_bytes[:4], "little") % (2**31))
        probs = np.random.dirichlet(np.ones(N_CLASSES) * 0.5).astype(np.float32)
        probs = np.clip(probs * N_CLASSES * 0.3, 0.0, 1.0)

    probs = np.clip(probs, 0.0, 1.0)

    # Ndërto response
    aneurysm_prob      = float(probs[-1])         # Indeksi 13 = Aneurysm Present
    aneurysm_present   = aneurysm_prob >= 0.5
    detected_locations = [
        LABEL_COLS[i]
        for i in range(N_CLASSES - 1)              # Shko tek 13 lokacionet (jo "Aneurysm Present")
        if float(probs[i]) >= 0.5
    ]

    prob_dict = {LABEL_COLS[i]: round(float(probs[i]), 4) for i in range(N_CLASSES)}

    return jsonify({
        "aneurysm_present":    aneurysm_present,
        "confidence":          round(aneurysm_prob * 100, 2),
        "probabilities":       prob_dict,
        "detected_locations":  detected_locations,
        "n_locations_detected":len(detected_locations),
        "model":               "CNN Baseline",
        "n_outputs":           N_CLASSES,
        "checkpoint_used":     _model_loaded_from_checkpoint,
        "status":              "success",
    })


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4: /predict_dicom — Inference DICOM (formati RSNA)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/predict_dicom", methods=["POST"])
def predict_dicom_endpoint():
    """
    Inference mbi seri DICOM — formati zyrtar RSNA.

    Request JSON:
        {"series_path": "/path/to/dicom/series/"}

    Response JSON:
        {
            "SeriesInstanceUID": str,
            "probabilities": {label: prob, ...},
            "aneurysm_present": bool,
            "status": "success"
        }

    Kërkon predict.py dhe pydicom të instaluara.
    """
    data = request.get_json(silent=True) or {}
    series_path = data.get("series_path", "")

    if not series_path:
        return jsonify({"error": "Mungon 'series_path' në request body."}), 400

    if not os.path.isdir(series_path):
        return jsonify({"error": f"Shtegu nuk ekziston: {series_path}"}), 400

    try:
        from predict import predict as rsna_predict  # type: ignore
        result_df   = rsna_predict(series_path)
        series_id   = os.path.basename(series_path.rstrip("/\\"))

        # Konverto DataFrame → dict
        if hasattr(result_df, "to_pandas"):
            probs_row = result_df.to_pandas().iloc[0]
        else:
            probs_row = result_df.iloc[0]

        prob_dict = {col: round(float(probs_row[col]), 4) for col in LABEL_COLS}
        aneurysm_present = prob_dict.get("Aneurysm Present", 0.5) >= 0.5

        return jsonify({
            "SeriesInstanceUID": series_id,
            "probabilities":     prob_dict,
            "aneurysm_present":  aneurysm_present,
            "status":            "success",
        })

    except ImportError:
        return jsonify({"error": "predict.py ose pydicom nuk disponueshëm."}), 500
    except Exception as e:
        return jsonify({"error": f"Gabim inference: {str(e)}"}), 500


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  NeuroVision AI — Flask Inference Server")
    print("  Blina Sopjani | ID: 69401 | Universum College")
    print("=" * 60)
    print(f"\n  Model    : ResNet-101 (14 outputs multi-label)")
    print(f"  Device   : {DEVICE}")
    print(f"  Checkpoint: {'✅ ngarkuar' if _model_loaded_from_checkpoint else '⚠  nuk u gjet (demo mode)'}")
    print(f"\n  Endpoints:")
    print(f"    GET  /health          — Status check")
    print(f"    GET  /model_info      — Info mbi modelin")
    print(f"    POST /predict         — Inference (imazh upload)")
    print(f"    POST /predict_dicom   — Inference DICOM (formati RSNA)")
    print(f"\n  Duke startuar: http://localhost:5005\n")

    app.run(host="0.0.0.0", port=5005, debug=False)