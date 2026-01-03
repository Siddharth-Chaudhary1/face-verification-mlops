# api/inference.py
import onnxruntime as ort
import numpy as np
from api.preprocess import preprocess

class FaceEmbeddingModel:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def extract_embedding(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Input: raw OpenCV image (H,W,3) BGR
        Output: L2-normalized embedding (512,)
        """
        x = preprocess(image_bgr)              # (3,112,112)
        x = np.expand_dims(x, axis=0)           # (1,3,112,112)

        embedding = self.session.run(
            None,
            {self.input_name: x}
        )[0]

        embedding = embedding.squeeze()

        # IMPORTANT: normalize embedding
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise ValueError("Zero-norm embedding")

        return embedding / norm
