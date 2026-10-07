"""
SkyShield - Quick Image Prediction Script
==========================================
Usage:
    python predict_image.py path/to/your/image.jpg
    python predict_image.py "C:/Users/you/Downloads/drone_photo.jpg"

What it does:
    Loads the saved MobileNetV3-Small model and predicts whether
    the image contains a Bird, Drone, or Other (aircraft/helicopter).

IMPORTANT - Read This First:
    Our model was trained on AERIAL/OVERHEAD view images of objects.
    Best results = images of flying objects from below or at similar angle.
    Random photos of drones on shelves or birds in cages will confuse it!
    But go ahead and test - that's the whole point!
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# -----------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------
MODEL_PATH   = "models_store/cnn_mobilenet_v3.pt"
CLASS_NAMES  = ["Bird (0)", "Drone (1)", "Other (2)"]
IMG_SIZE     = 224
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calibration settings (from Notebook 05)
TEMPERATURE      = 0.6426   # Temperature scaling factor
DRONE_THRESHOLD  = 0.350    # If drone prob > this -> DRONE ALERT
REJECT_THRESHOLD = 0.550    # If max prob < this -> UNIDENTIFIED

# -----------------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------------
def load_model():
    print("[*] Loading MobileNetV3-Small model...")
    model = models.mobilenet_v3_small(weights=None)

    # Reconstruct the exact custom classifier used during training:
    # classifier.1 = Linear(576->128), classifier.4 = Linear(128->3)
    in_features = 576  # MobileNetV3-Small feature extractor output
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(in_features, 128),
        torch.nn.Hardswish(),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(128, 3),
    )

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    print("    [OK] Model loaded on %s\n" % DEVICE.type.upper())
    return model


# -----------------------------------------------------------------
# PREPROCESS IMAGE
# -----------------------------------------------------------------
def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0), img


# -----------------------------------------------------------------
# PREDICT
# -----------------------------------------------------------------
def predict(model, image_path):
    tensor, _ = preprocess(image_path)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        scaled = logits / TEMPERATURE
        probs  = torch.softmax(scaled, dim=1).cpu().numpy()[0]

    bird_p, drone_p, other_p = probs

    if drone_p >= DRONE_THRESHOLD:
        decision       = ">>> DRONE ALERT <<<"
        decision_class = "Drone (1)"
    elif float(max(probs)) < REJECT_THRESHOLD:
        decision       = "??? UNIDENTIFIED (low confidence)"
        decision_class = "UNIDENTIFIED"
    else:
        pred_idx = int(np.argmax(probs))
        decision_class = CLASS_NAMES[pred_idx]
        if pred_idx == 0:
            decision = "BIRD -- No threat"
        else:
            decision = "OTHER (aircraft/helicopter)"

    return {
        "decision":   decision,
        "class":      decision_class,
        "bird_prob":  float(bird_p),
        "drone_prob": float(drone_p),
        "other_prob": float(other_p),
        "max_prob":   float(max(probs)),
    }


# -----------------------------------------------------------------
# PRETTY PRINT
# -----------------------------------------------------------------
def print_result(result, image_path):
    fname = os.path.basename(image_path)
    print("=" * 55)
    print("  SkyShield Vision Prediction")
    print("  Image : %s" % fname)
    print("=" * 55)
    print("\n  RESULT --> %s\n" % result['decision'])
    print("  -- Probability Breakdown --")
    bar_b = "#" * int(result['bird_prob']  * 30)
    bar_d = "#" * int(result['drone_prob'] * 30)
    bar_o = "#" * int(result['other_prob'] * 30)
    print("  Bird  : %5.1f%%  |%-30s|" % (result['bird_prob']  * 100, bar_b))
    print("  Drone : %5.1f%%  |%-30s|" % (result['drone_prob'] * 100, bar_d))
    print("  Other : %5.1f%%  |%-30s|" % (result['other_prob'] * 100, bar_o))
    print()
    print("  Drone Alert Threshold : %.0f%%" % (DRONE_THRESHOLD  * 100))
    print("  Reject Threshold      : %.0f%%" % (REJECT_THRESHOLD * 100))
    print("  Temperature Scaling T : %.4f"   %  TEMPERATURE)
    print("=" * 55)

    if result['max_prob'] < 0.55:
        print("\n  [WARNING] LOW CONFIDENCE -- Image may be very different")
        print("     from aerial training data (overhead/flying view).")
        print("     Try a clearer aerial photo for better results.\n")


# -----------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("\n[ERROR] No image path provided!")
        print("   Usage: python predict_image.py path/to/image.jpg\n")
        sys.exit(1)

    image_path = sys.argv[1].strip('"').strip("'")

    if not os.path.isfile(image_path):
        print("\n[ERROR] File not found: %s\n" % image_path)
        sys.exit(1)

    model  = load_model()
    result = predict(model, image_path)
    print_result(result, image_path)


if __name__ == "__main__":
    main()
