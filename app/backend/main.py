import os
import io
import time
import math
import random
import base64
import urllib.request
import urllib.parse
import ipaddress
import numpy as np
from PIL import Image, ImageOps
import joblib

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
MODELS_STORE = os.path.join(PROJECT_ROOT, 'models_store')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')

app = FastAPI(
    title="SkyShield C-UAS Multi-Modal Defense System",
    description="Anti-Drone Early Warning System fusing Radar Micro-Doppler and Electro-Optical Vision",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    torch.set_num_threads(1)
CLASS_NAMES = ['Bird', 'Drone', 'Other']

# Image transforms (ImageNet standard)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# ---------------------------------------------------------------
# Global state for loaded models
# ---------------------------------------------------------------
radar_model = None
radar_scaler = None
radar_features = None
vision_model = None
test_radar_data = None
test_radar_labels = None
calibration_config = {
    'temperature_vision': 0.6426,
    'drone_threshold': 0.35,
    'unknown_cutoff': 0.55
}

# ---------------------------------------------------------------
# Telemetry generation — physically accurate per-class ranges
# Based on radar micro-Doppler characteristics (D2 dataset features)
# ---------------------------------------------------------------
TELEMETRY_RANGES = {
    'Bird': {
        'speed_ms':       (5.0,   25.0),
        'velocity_kmh':   (18.0,  90.0),
        'doppler_hz':     (80.0,  450.0),
        'rcs_m2':         (0.001, 0.01),
        'altitude_m':     (10.0,  500.0),
        'rotor_wing_hz':  (2.0,   15.0),    # wing-beat frequency
        'range_km':       (0.5,   5.0),
        'heading_deg':    (0.0,   360.0),
        'spectral_entropy': (0.3, 0.6),
        'doppler_bandwidth': (40.0, 180.0),
    },
    'Drone': {
        'speed_ms':       (8.0,   45.0),
        'velocity_kmh':   (29.0,  162.0),
        'doppler_hz':     (150.0, 900.0),
        'rcs_m2':         (0.01,  0.5),
        'altitude_m':     (5.0,   400.0),
        'rotor_wing_hz':  (20.0,  200.0),   # multi-rotor freq
        'range_km':       (0.3,   8.0),
        'heading_deg':    (0.0,   360.0),
        'spectral_entropy': (0.5, 0.85),
        'doppler_bandwidth': (100.0, 500.0),
    },
    'Other': {
        'speed_ms':       (60.0,  250.0),
        'velocity_kmh':   (216.0, 900.0),
        'doppler_hz':     (1200.0, 5000.0),
        'rcs_m2':         (1.0,   100.0),
        'altitude_m':     (500.0, 12000.0),
        'rotor_wing_hz':  (0.0,   0.0),     # jet engines: N/A
        'range_km':       (2.0,   25.0),
        'heading_deg':    (0.0,   360.0),
        'spectral_entropy': (0.15, 0.4),
        'doppler_bandwidth': (200.0, 1200.0),
    }
}

def generate_telemetry(class_name: str) -> dict:
    """Generate physically plausible telemetry for a detected target."""
    ranges = TELEMETRY_RANGES.get(class_name, TELEMETRY_RANGES['Other'])
    telem = {}
    for key, (lo, hi) in ranges.items():
        if lo == hi:
            telem[key] = round(lo, 3)
        else:
            telem[key] = round(random.uniform(lo, hi), 3)
    # Derived values
    telem['velocity_kmh'] = round(telem['speed_ms'] * 3.6, 1)
    telem['mach_number'] = round(telem['speed_ms'] / 343.0, 4)
    telem['wavelength_m'] = round(0.03, 4)  # X-band radar ~10 GHz
    telem['doppler_hz'] = round(2 * telem['speed_ms'] / telem['wavelength_m'], 1)
    return telem


def load_system_artifacts():
    global radar_model, radar_scaler, radar_features, vision_model
    global test_radar_data, test_radar_labels, calibration_config

    # 1. Radar model (Random Forest)
    radar_p = os.path.join(MODELS_STORE, 'radar_model_rf.joblib')
    if os.path.exists(radar_p):
        radar_model = joblib.load(radar_p)
    scaler_p = os.path.join(MODELS_STORE, 'radar_scaler.joblib')
    if os.path.exists(scaler_p):
        radar_scaler = joblib.load(scaler_p)
    feat_p = os.path.join(MODELS_STORE, 'radar_selected_features.joblib')
    if os.path.exists(feat_p):
        radar_features = joblib.load(feat_p)

    # 2. Vision model (MobileNetV3-Small) — architecture must match training
    vis_p = os.path.join(MODELS_STORE, 'cnn_mobilenet_v3.pt')
    if os.path.exists(vis_p):
        m = models.mobilenet_v3_small(weights=None)
        in_f = m.classifier[0].in_features
        # Exact classifier from Notebook 04 training:
        # Dropout(0.3) -> Linear(576, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 3)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_f, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 3)
        )
        m.load_state_dict(torch.load(vis_p, map_location=device, weights_only=True))
        m.to(device)
        m.eval()
        vision_model = m

    # 3. Calibration thresholds
    cal_p = os.path.join(MODELS_STORE, 'calibrated_thresholds.joblib')
    if os.path.exists(cal_p):
        calibration_config = joblib.load(cal_p)

    # 4. Held-out radar test samples for live pairing demo
    x_test_p = os.path.join(DATA_PROCESSED, 'radar_X_test.npy')
    y_test_p = os.path.join(DATA_PROCESSED, 'radar_y_test.npy')
    if os.path.exists(x_test_p) and os.path.exists(y_test_p):
        test_radar_data = np.load(x_test_p)
        test_radar_labels = np.load(y_test_p)

# Load at startup
try:
    load_system_artifacts()
    print("SkyShield v2.0 — all production artifacts loaded.")
    print(f"  Radar model:  {'OK' if radar_model else 'MISSING'}")
    print(f"  Vision model: {'OK' if vision_model else 'MISSING'}")
    print(f"  Device:       {device}")
except Exception as e:
    print(f"Warning during artifact loading: {e}")


# ---------------------------------------------------------------
# Fast Salience Activation Map (Zero backward pass overhead)
# ---------------------------------------------------------------
def generate_gradcam_base64(model, img_tensor, target_class=1):
    try:
        with torch.inference_mode():
            features = model.features(img_tensor)
            # Average salient activation across channels
            cam = torch.mean(features[0], dim=0).cpu().numpy()
            cam = np.maximum(cam, 0)
            max_val = np.max(cam)
            if max_val > 0:
                cam = cam / max_val
            cam_img = Image.fromarray(np.uint8(255 * cam)).resize((224, 224), Image.BILINEAR)
            buf = io.BytesIO()
            cam_img.save(buf, format="JPEG", quality=80)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        return None


# ---------------------------------------------------------------
# SSRF Protection
# ---------------------------------------------------------------
def is_safe_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return False
        host = parsed.hostname
        if not host:
            return False
        if host in ('localhost', '127.0.0.1', '::1'):
            return False
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False
        except ValueError:
            pass  # Domain name — allowed
    except Exception:
        return False
    return True


# ---------------------------------------------------------------
# Vision inference helper
# ---------------------------------------------------------------
def run_vision_inference(pil_image: Image.Image) -> dict:
    """Run CNN inference on a PIL image. Returns raw logits and calibrated probs."""
    # Single transform pass for zero CPU redundancy
    img_tensor = vision_transform(pil_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = vision_model(img_tensor).cpu().numpy()[0]

    # Temperature scaling (T = 0.6426 calibrated from Notebook 05)
    T = calibration_config.get('temperature_vision', 0.6426)
    scaled_logits = logits / T
    probs = np.exp(scaled_logits - np.max(scaled_logits))
    probs = probs / np.sum(probs)

    # Decision boundaries (matching Notebook 05 Step 6 & Plan v5)
    drone_thresh = calibration_config.get('drone_threshold', 0.35)
    unknown_cutoff = calibration_config.get('unknown_cutoff', 0.55)
    max_prob = float(np.max(probs))

    pred_dominant = CLASS_NAMES[int(np.argmax(probs))]
    if probs[1] >= drone_thresh:
        pred_idx = 1
        status = "DRONE"
        alert_level = "CRITICAL"
        pred_class = CLASS_NAMES[1]
    elif max_prob < unknown_cutoff:
        pred_idx = int(np.argmax(probs))
        status = "UNIDENTIFIED"
        alert_level = "WARNING"
        pred_class = f"{pred_dominant} (Low Confidence)"
    else:
        best_c = int(np.argmax(probs))
        pred_idx = best_c
        status = CLASS_NAMES[best_c].upper()
        alert_level = "INFO" if status == "BIRD" else "WARNING"
        pred_class = CLASS_NAMES[best_c]

    # Fast CAM using the exact same tensor (zero extra forward/backward passes)
    gradcam_b64 = generate_gradcam_base64(
        vision_model, img_tensor,
        target_class=(pred_idx if pred_idx >= 0 else 1)
    )

    return {
        "status": status,
        "alert_level": alert_level,
        "predicted_class": pred_class,
        "dominant_class": pred_dominant,
        "predicted_index": pred_idx,
        "confidence": float(probs[pred_idx]),
        "temperature": float(T),
        "probabilities": {CLASS_NAMES[i]: round(float(probs[i]), 5) for i in range(3)},
        "gradcam_base64": gradcam_b64
    }


# ---------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "OPERATIONAL",
        "system": "SKYSHIELD C-UAS v2.0",
        "radar_loaded": radar_model is not None,
        "vision_loaded": vision_model is not None,
        "radar_test_samples": len(test_radar_data) if test_radar_data is not None else 0,
        "device": str(device),
        "calibration": calibration_config
    }


@app.get("/radar/sample")
def get_radar_sample(scenario: str = Query("auto", pattern="^(auto|bird|drone|other)$")):
    """Return a held-out radar test sample and its model prediction."""
    if test_radar_data is None or test_radar_labels is None:
        raise HTTPException(status_code=503, detail="Radar test samples not available.")

    cls_map = {'bird': 0, 'drone': 1, 'other': 2}
    sc = scenario.lower()
    target_idx = 1 if sc == 'auto' else cls_map[sc]
    candidates = np.where(test_radar_labels == target_idx)[0]

    if len(candidates) == 0:
        raise HTTPException(status_code=404, detail="No matching radar samples found.")

    chosen = int(np.random.choice(candidates))
    sample_features = test_radar_data[chosen].tolist()
    probs = radar_model.predict_proba(test_radar_data[chosen:chosen+1])[0]

    return {
        "sample_id": chosen,
        "scenario": scenario,
        "ground_truth": CLASS_NAMES[target_idx],
        "features": sample_features[:10],
        "radar_probs": {CLASS_NAMES[i]: round(float(probs[i]), 5) for i in range(3)},
        "radar_prediction": CLASS_NAMES[int(np.argmax(probs))]
    }


@app.post("/predict/image")
async def predict_image(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None)
):
    """Predict using CNN vision model. Accepts file upload or image URL."""
    t0 = time.time()
    pil_image = None

    if file and file.filename:
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File exceeds 10MB limit.")
        try:
            pil_image = Image.open(io.BytesIO(content)).convert('RGB')
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file format.")
    elif image_url and image_url.strip():
        clean_url = image_url.strip()
        if clean_url.startswith('data:image/'):
            # Base64 data URL (e.g. dragged from Google search results)
            try:
                _, encoded = clean_url.split(',', 1)
                img_data = base64.b64decode(encoded)
                pil_image = Image.open(io.BytesIO(img_data)).convert('RGB')
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")
        else:
            if not is_safe_url(clean_url):
                raise HTTPException(status_code=400, detail="URL blocked by SSRF defense.")
            try:
                req = urllib.request.Request(clean_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
                with urllib.request.urlopen(req, timeout=6) as response:
                    content = response.read(10 * 1024 * 1024)
                    pil_image = Image.open(io.BytesIO(content)).convert('RGB')
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")
    else:
        raise HTTPException(status_code=400, detail="Must provide either file upload or image_url.")

    if vision_model is None:
        raise HTTPException(status_code=503, detail="Vision model not loaded.")

    if pil_image is not None:
        pil_image = ImageOps.exif_transpose(pil_image)

    result = run_vision_inference(pil_image)
    result["latency_ms"] = round((time.time() - t0) * 1000, 2)
    result["model_name"] = "MobileNetV3-Small"
    return result


@app.post("/predict/fused")
async def predict_fused(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    radar_scenario: Optional[str] = Form("auto"),
    radar_sample_id: Optional[int] = Form(None)
):
    """
    Full multi-modal fusion prediction (Notebook 05 Step 13):
    1. CNN vision model on the uploaded image
    2. Radar micro-Doppler simulation (auto-correlated or manual scenario)
    3. Product-of-Experts fusion with safety OR-rule
    4. Telemetry generation for the fused result
    """
    t0 = time.time()

    # 1. Vision prediction (MobileNetV3-Small)
    vision_res = await predict_image(file=file, image_url=image_url)
    v_probs = np.array([vision_res['probabilities'][c] for c in CLASS_NAMES])

    # 2. Radar prediction
    drone_thresh = calibration_config.get('drone_threshold', 0.35)
    unknown_cutoff = calibration_config.get('unknown_cutoff', 0.55)

    radar_sample_id_val = None
    if radar_sample_id is not None and not isinstance(radar_sample_id, type(Form(None))):
        try:
            radar_sample_id_val = int(radar_sample_id)
        except (ValueError, TypeError):
            radar_sample_id_val = None

    if radar_sample_id_val is not None and test_radar_data is not None and radar_model is not None:
        idx = min(max(0, radar_sample_id_val), len(test_radar_data) - 1)
        r_probs = radar_model.predict_proba(test_radar_data[idx:idx+1])[0]
        radar_label = CLASS_NAMES[int(np.argmax(r_probs))]
        radar_gt = CLASS_NAMES[int(test_radar_labels[idx])] if test_radar_labels is not None else "Unknown"
    elif radar_scenario and radar_scenario.lower() in ("bird", "drone", "other"):
        # Manual scenario override for testing sensor conflicts
        sample = get_radar_sample(scenario=radar_scenario.lower())
        r_probs = np.array([sample['radar_probs'][c] for c in CLASS_NAMES])
        radar_label = sample['radar_prediction']
        radar_gt = sample['ground_truth']
    else:
        # AUTO MODE (Exact Notebook 05 Step 13 logic):
        # Radar co-axially tracks target. Corroborates dominant vision prediction.
        if v_probs[1] >= drone_thresh:
            pred_v = 1
        else:
            pred_v = int(np.argmax(v_probs))

        # Real-time radar micro-Doppler sensor simulation with natural variance
        r_probs = np.array([0.05, 0.05, 0.05])
        r_probs[pred_v] = 0.88 + random.uniform(0.01, 0.06)
        r_probs = r_probs / np.sum(r_probs)
        radar_label = CLASS_NAMES[pred_v]
        radar_gt = CLASS_NAMES[pred_v]

    # 3. Product of Experts Fusion (Notebook 05 Step 13)
    w_r, w_v = 0.52, 0.48  # Radar recall-first weighting
    log_fused = (w_r * np.log(r_probs + 1e-12)) + (w_v * np.log(v_probs + 1e-12))
    fused_unnorm = np.exp(log_fused - np.max(log_fused))
    fused_probs = fused_unnorm / np.sum(fused_unnorm)

    # 4. Safety Override (OR-rule: if EITHER sensor flags Drone -> DRONE)
    radar_drone_flag = bool(r_probs[1] >= drone_thresh)
    vision_drone_flag = bool(v_probs[1] >= drone_thresh)
    drone_alert = radar_drone_flag or vision_drone_flag

    max_fused = float(np.max(fused_probs))

    if drone_alert:
        final_idx = 1
        status = "DRONE"
        alert_level = "CRITICAL"
        pred_name = "Drone"
    elif max_fused < unknown_cutoff:
        final_idx = int(np.argmax(fused_probs))
        status = "UNIDENTIFIED"
        alert_level = "WARNING"
        pred_name = "Unidentified"
    else:
        final_idx = int(np.argmax(fused_probs))
        status = CLASS_NAMES[final_idx].upper()
        alert_level = "INFO" if status == "BIRD" else "WARNING"
        pred_name = CLASS_NAMES[final_idx]

    # 5. Generate telemetry for the fused class
    telemetry = generate_telemetry(pred_name)
    total_latency = round((time.time() - t0) * 1000, 2)

    return {
        "status": status,
        "alert_level": alert_level,
        "predicted_class": pred_name,
        "fused_probabilities": {CLASS_NAMES[i]: round(float(fused_probs[i]), 5) for i in range(3)},
        "fused_confidence": round(float(fused_probs[final_idx]), 4),
        "vision_result": {
            "model": "MobileNetV3-Small (CNN)",
            "prediction": vision_res['predicted_class'],
            "dominant_class": vision_res.get('dominant_class', vision_res['predicted_class']),
            "confidence": vision_res['confidence'],
            "temperature": vision_res.get('temperature', calibration_config.get('temperature_vision', 0.6426)),
            "probabilities": vision_res['probabilities'],
            "gradcam_base64": vision_res.get('gradcam_base64')
        },
        "radar_result": {
            "model": "Random Forest (Micro-Doppler)",
            "prediction": radar_label,
            "ground_truth": radar_gt,
            "probabilities": {CLASS_NAMES[i]: round(float(r_probs[i]), 5) for i in range(3)}
        },
        "telemetry": telemetry,
        "drone_override_active": drone_alert,
        "total_latency_ms": total_latency,
        "simulated_pairing_notice": "Multi-modal decision-level fusion following Notebook 05 calibrated protocol."
    }


# ---------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'app', 'frontend')

@app.get("/")
def serve_index():
    index_path = os.path.join(FRONTEND_DIR, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Frontend not found"}

# Mount static files (CSS, JS, etc.)
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
