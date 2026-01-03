# api/preprocess.py
import cv2
import numpy as np

INPUT_SIZE = (112, 112)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("Invalid image")

    image = cv2.resize(image_bgr, INPUT_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    image = np.transpose(image, (2, 0, 1))  # CHW

    return image
