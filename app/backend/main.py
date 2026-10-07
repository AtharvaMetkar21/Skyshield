import os
import io
import time
import base64
import urllib.request
import urllib.parse
import ipaddress
import numpy as np
from PIL import Image
import joblib

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Bird', 'Drone', 'Other']

# Image transforms
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Global state
radar_model = None
radar_scaler = None
radar_features = None
vision_model = None
test_radar_data = None
test_radar_labels = None
calibration_config = {
    'temperature_vision': 1.15,
    'drone_threshold': 0.35,
    'unknown_cutoff': 0.55
}

def load_system_artifacts():
    global radar_model, radar_scaler, radar_features, vision_model
    global test_radar_data, test_radar_labels, calibration_config
    
    # 1. Radar
    radar_p = os.path.join(MODELS_STORE, 'radar_model_rf.joblib')
    if os.path.exists(radar_p):
        radar_model = joblib.load(radar_p)
    scaler_p = os.path.join(MODELS_STORE, 'radar_scaler.joblib')
    if os.path.exists(scaler_p):
        radar_scaler = joblib.load(scaler_p)
    feat_p = os.path.join(MODELS_STORE, 'radar_selected_features.joblib')
    if os.path.exists(feat_p):
        radar_features = joblib.load(feat_p)
        
    # 2. Vision Model
    vis_p = os.path.join(MODELS_STORE, 'cnn_mobilenet_v3.pt')
    if os.path.exists(vis_p):
        m = models.mobilenet_v3_small(weights=None)
        in_f = m.classifier[0].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_f, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 3)
        )
        m.load_state_dict(torch.load(vis_p, map_location=device))
        m.to(device)
        m.eval()
        vision_model = m

    # 3. Calibration
    cal_p = os.path.join(MODELS_STORE, 'calibrated_thresholds.joblib')
    if os.path.exists(cal_p):
        calibration_config = joblib.load(cal_p)
        
    # 4. Held-out Radar Test Samples for live demo
    x_test_p = os.path.join(DATA_PROCESSED, 'radar_X_test.npy')
    y_test_p = os.path.join(DATA_PROCESSED, 'radar_y_test.npy')
    if os.path.exists(x_test_p) and os.path.exists(y_test_p):
        test_radar_data = np.load(x_test_p)
        test_radar_labels = np.load(y_test_p)

# Load at startup
try:
    load_system_artifacts()
    print("SkyShield production artifacts loaded successfully.")
except Exception as e:
    print(f"Warning during artifact loading: {e}")

# Helper: Grad-CAM generation
def generate_gradcam_base64(model, img_tensor, target_class):
    try:
        target_layer = model.features[-1]
        activations = None
        gradients = None
        
        def forward_hook(module, input, output):
            nonlocal activations
            activations = output
            
        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0]
            
        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)
        
        model.eval()
        output = model(img_tensor)
        model.zero_grad()
        output[0, target_class].backward()
        
        h1.remove()
        h2.remove()
        
        grads = gradients[0].cpu().data.numpy()
        acts = activations[0].cpu().data.numpy()
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
            
        cam_img = Image.fromarray(np.uint8(255 * cam)).resize((224, 224), Image.BILINEAR)
        buf = io.BytesIO()
        cam_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        return None

# SSRF Protection Validator
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
        # Check IP ranges
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            return False
    except ValueError:
        # Not a raw IP, domain name passes
        pass
    except Exception:
        return False
    return True

@app.get("/health")
def health_check():
    return {
        "status": "HEALTHY",
        "radar_loaded": radar_model is not None,
        "vision_loaded": vision_model is not None,
        "device": str(device)
    }

@app.get("/radar/sample")
def get_radar_sample(scenario: str = Query("drone", regex="^(bird|drone|other)$")):
    if test_radar_data is None or test_radar_labels is None:
        raise HTTPException(status_code=503, detail="Radar test samples not available.")
        
    cls_map = {'bird': 0, 'drone': 1, 'other': 2}
    target_idx = cls_map[scenario.lower()]
    candidates = np.where(test_radar_labels == target_idx)[0]
    
    if len(candidates) == 0:
        raise HTTPException(status_code=404, detail="No matching radar samples found.")
        
    chosen = np.random.choice(candidates)
    sample_features = test_radar_data[chosen].tolist()
    
    # Get model prediction
    probs = radar_model.predict_proba(test_radar_data[chosen:chosen+1])[0]
    
    return {
        "sample_id": int(chosen),
        "scenario": scenario,
        "ground_truth": CLASS_NAMES[target_idx],
        "features": sample_features[:10], # Return first 10 for display
        "radar_probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(3)},
        "radar_prediction": CLASS_NAMES[int(np.argmax(probs))]
    }

@app.post("/predict/image")
async def predict_image(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None)
):
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
    elif image_url:
        if not is_safe_url(image_url):
            raise HTTPException(status_code=400, detail="URL blocked by SSRF defense (private/loopback address).")
        try:
            req = urllib.request.Request(image_url, headers={'User-Agent': 'SkyShield/1.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                content = response.read(10 * 1024 * 1024)
                pil_image = Image.open(io.BytesIO(content)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {e}")
    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded image or image_url.")
        
    if vision_model is None:
        raise HTTPException(status_code=503, detail="Vision model not loaded.")
        
    # Preprocess
    img_tensor = vision_transform(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = vision_model(img_tensor).cpu().numpy()[0]
        
    # Temperature scaling
    T = calibration_config.get('temperature_vision', 1.15)
    scaled_logits = logits / T
    probs = np.exp(scaled_logits - np.max(scaled_logits))
    probs = probs / np.sum(probs)
    
    # Decision boundaries
    drone_thresh = calibration_config.get('drone_threshold', 0.35)
    unknown_tau = calibration_config.get('unknown_cutoff', 0.55)
    
    if probs[1] >= drone_thresh:
        pred_idx = 1
        status = "DRONE"
        alert_level = "CRITICAL"
    else:
        best_c = int(np.argmax(probs))
        conf = float(probs[best_c])
        if conf < unknown_tau:
            pred_idx = -1
            status = "UNIDENTIFIED"
            alert_level = "AMBER"
        else:
            pred_idx = best_c
            status = CLASS_NAMES[best_c].upper()
            alert_level = "INFO" if status == "BIRD" else "WARNING"
            
    latency_ms = (time.time() - t0) * 1000
    gradcam_b64 = generate_gradcam_base64(vision_model, img_tensor, target_class=(pred_idx if pred_idx >= 0 else 1))
    
    return {
        "status": status,
        "alert_level": alert_level,
        "predicted_class": CLASS_NAMES[pred_idx] if pred_idx >= 0 else "Unidentified",
        "confidence": float(probs[pred_idx]) if pred_idx >= 0 else float(np.max(probs)),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(3)},
        "latency_ms": round(latency_ms, 2),
        "gradcam_base64": gradcam_b64
    }

class FusedRequest(BaseModel):
    image_url: Optional[str] = None
    radar_scenario: Optional[str] = "drone"
    radar_sample_id: Optional[int] = None

@app.post("/predict/fused")
async def predict_fused(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    radar_scenario: Optional[str] = Form("drone"),
    radar_sample_id: Optional[int] = Form(None)
):
    # 1. Vision prediction
    vision_res = await predict_image(file=file, image_url=image_url)
    v_probs = np.array([vision_res['probabilities'][c] for c in CLASS_NAMES])
    
    # 2. Radar prediction
    if radar_sample_id is not None and test_radar_data is not None:
        idx = min(max(0, radar_sample_id), len(test_radar_data) - 1)
        r_probs = radar_model.predict_proba(test_radar_data[idx:idx+1])[0]
        radar_label = CLASS_NAMES[int(np.argmax(r_probs))]
    else:
        sample = get_radar_sample(scenario=radar_scenario)
        r_probs = np.array([sample['radar_probs'][c] for c in CLASS_NAMES])
        radar_label = sample['radar_prediction']
        
    # 3. Product of Experts Fusion
    w_r, w_v = 0.52, 0.48
    log_fused = (w_r * np.log(r_probs + 1e-12)) + (w_v * np.log(v_probs + 1e-12))
    fused_unnorm = np.exp(log_fused - np.max(log_fused))
    fused_probs = fused_unnorm / np.sum(fused_unnorm)
    
    # 4. Safety Override
    drone_thresh = calibration_config.get('drone_threshold', 0.35)
    unknown_tau = calibration_config.get('unknown_cutoff', 0.55)
    
    radar_drone_flag = bool(r_probs[1] >= drone_thresh)
    vision_drone_flag = bool(v_probs[1] >= drone_thresh)
    
    if radar_drone_flag or vision_drone_flag:
        status = "DRONE"
        alert_level = "CRITICAL"
        pred_name = "Drone"
    else:
        best_c = int(np.argmax(fused_probs))
        if fused_probs[best_c] < unknown_tau:
            status = "UNIDENTIFIED"
            alert_level = "AMBER"
            pred_name = "Unidentified"
        else:
            status = CLASS_NAMES[best_c].upper()
            alert_level = "INFO" if status == "BIRD" else "WARNING"
            pred_name = CLASS_NAMES[best_c]
            
    return {
        "status": status,
        "alert_level": alert_level,
        "predicted_class": pred_name,
        "fused_probabilities": {CLASS_NAMES[i]: float(fused_probs[i]) for i in range(3)},
        "vision_result": vision_res,
        "radar_result": {
            "prediction": radar_label,
            "probabilities": {CLASS_NAMES[i]: float(r_probs[i]) for i in range(3)}
        },
        "drone_override_active": radar_drone_flag or vision_drone_flag,
        "simulated_pairing_notice": "Unpaired sensor simulation: radar sample paired with visual feed for operational defense simulation."
    }

# Mount static frontend files if folder exists
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'app', 'frontend')
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
