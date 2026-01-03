# # api/main.py
# from fastapi import FastAPI, HTTPException
# import base64
# import cv2
# import numpy as np

# from api.inference import FaceEmbeddingModel
# from api.similarity import cosine_similarity

# app = FastAPI(title="Face Matching Service")

# model = FaceEmbeddingModel("model/facedet.onnx")

# def decode_image(b64: str) -> np.ndarray:
#     try:
#         img_bytes = base64.b64decode(b64)
#         arr = np.frombuffer(img_bytes, np.uint8)
#         img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#         if img is None:
#             raise ValueError
#         return img
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid image input")

# @app.get("/health")
# def health():
#     return {"status": "ok"}

# @app.post("/compare")
# def compare_faces(
#     image1: str,
#     image2: str,
#     threshold: float = 0.35
# ):
#     img1 = decode_image(image1)
#     img2 = decode_image(image2)

#     emb1 = model.extract_embedding(img1)
#     emb2 = model.extract_embedding(img2)

#     score = cosine_similarity(emb1, emb2)

#     return {
#         "similarity": score,
#         "is_match": score > threshold,
#         "threshold": threshold
#     }

from fastapi import FastAPI, HTTPException
import base64
import cv2
import numpy as np

from api.inference import FaceEmbeddingModel
from api.similarity import cosine_similarity
from api.schemas import CompareRequest

app = FastAPI(title="Face Matching Service")

model = FaceEmbeddingModel("model/facedet.onnx")

def decode_image(b64: str) -> np.ndarray:
    try:
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError
        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image input")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/compare")
def compare_faces(req: CompareRequest):
    img1 = decode_image(req.image1)
    img2 = decode_image(req.image2)

    emb1 = model.extract_embedding(img1)
    emb2 = model.extract_embedding(img2)

    score = cosine_similarity(emb1, emb2)

    return {
        "similarity": score,
        "is_match": score > req.threshold,
        "threshold": req.threshold
    }
