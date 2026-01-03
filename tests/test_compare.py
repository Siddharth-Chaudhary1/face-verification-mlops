import base64
import requests

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

img1 = encode_image("p1_1.jpg")
img2 = encode_image("p2_2.jpg")

payload = {
    "image1": img1,
    "image2": img2,
    "threshold": 0.35
}

resp = requests.post(
    "http://127.0.0.1:8000/compare",
    json=payload
)

print(resp.status_code)
print(resp.json())
