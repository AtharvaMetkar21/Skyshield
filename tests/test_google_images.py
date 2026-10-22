import os
import urllib.request
import json

def test_google_images():
    boundary = "----WebKitFormBoundaryGoogleTest"
    test_files = [
        r"C:\Users\athar\Downloads\images (8).jpg",
        r"C:\Users\athar\Downloads\birds.jpg",
        r"C:\Users\athar\Downloads\stress_Drone_4.jpg",
        r"C:\Users\athar\Downloads\stress_Bird_0.jpg",
        r"C:\Users\athar\Downloads\images (1).jpg",
        r"C:\Users\athar\Downloads\images (2).jpg",
        r"C:\Users\athar\Downloads\images (3).jpg",
        r"C:\Users\athar\Downloads\images (4).jpg",
        r"C:\Users\athar\Downloads\images (5).jpg",
        r"C:\Users\athar\Downloads\images (6).jpg",
        r"C:\Users\athar\Downloads\images (7).jpg",
        r"C:\Users\athar\Downloads\images (9).jpg",
        r"C:\Users\athar\Downloads\images (10).jpg",
        r"data\raw\internet_stress_set\stress_Drone_1.jpg",
        r"data\raw\internet_stress_set\stress_Drone_3.jpg",
        r"data\raw\internet_stress_set\stress_Bird_7.jpg",
        r"data\raw\internet_stress_set\stress_Other_5.jpg",
        r"data\raw\internet_stress_set\stress_Other_11.jpg"
    ]

    print("=" * 85)
    print("        SKYSHIELD C-UAS — REAL GOOGLE & INTERNET IMAGE VERIFICATION")
    print("=" * 85)

    for p in test_files:
        if not os.path.exists(p):
            continue
        try:
            with open(p, "rb") as f:
                img_bytes = f.read()

            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(p)}"\r\n'
                f"Content-Type: image/jpeg\r\n\r\n"
            ).encode("latin1") + img_bytes + (
                f"\r\n--{boundary}\r\n"
                f'Content-Disposition: form-data; name="radar_scenario"\r\n\r\nauto\r\n--{boundary}--\r\n'
            ).encode("latin1")

            req = urllib.request.Request(
                "http://127.0.0.1:8000/predict/fused",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
                v = data["vision_result"]
                r = data["radar_result"]
                pred = data["predicted_class"]
                alert = data["alert_level"]
                conf = data["fused_confidence"] * 100
                v_conf = v["confidence"] * 100
                print(f"[{pred:5}] ({conf:5.1f}%) | Vision: {v['prediction']:5} ({v_conf:5.1f}%) | Radar: {r['prediction']:5} | Alert: {alert:8} | File: {os.path.basename(p)}")
        except Exception as e:
            print(f"Error on {os.path.basename(p)}: {e}")

    print("=" * 85)

if __name__ == "__main__":
    test_google_images()
