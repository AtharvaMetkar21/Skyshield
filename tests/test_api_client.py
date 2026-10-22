import urllib.request
import json

def test_api():
    boundary = "----WebKitFormBoundaryX"
    for img_path, target_scenario in [
        (r"C:\Users\athar\Downloads\images (8).jpg", "auto"),
        ("data/interim/cropped_aod4/aod4_drone_2102.jpg", "auto"),
        ("data/interim/cropped_aod4/aod4_bird_0.jpg", "auto"),
        ("data/interim/cropped_aod4/aod4_airplane_4174.jpg", "auto"),
    ]:
        print(f"\n--- Testing {img_path} with radar_scenario={target_scenario} ---")
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode("latin1") + img_bytes + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="radar_scenario"\r\n\r\n'
            f"{target_scenario}"
            f"\r\n--{boundary}--\r\n"
        ).encode("latin1")

        req = urllib.request.Request(
            "http://127.0.0.1:8000/predict/fused",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print("STATUS:", data.get("status"))
            print("PREDICTED CLASS:", data.get("predicted_class"))
            print("FUSED PROBS:", data.get("fused_probabilities"))
            print("VISION RES:", data.get("vision_result")["prediction"], data.get("vision_result")["confidence"])
            print("RADAR RES:", data.get("radar_result")["prediction"], data.get("radar_result")["probabilities"])

if __name__ == "__main__":
    test_api()
