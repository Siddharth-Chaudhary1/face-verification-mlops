# Face Verification System (MLOps)

An end-to-end face verification system built with ONNX Runtime and FastAPI, focusing on correct training–serving parity, deployment, and model lifecycle management.

## Features
- ONNX-based inference (framework-agnostic)
- FastAPI backend with robust input validation
- Dockerized deployment
- MLflow model registry and experiment tracking
- Streamlit frontend for real-time demo
- Cosine similarity–based face matching

## Architecture
Image → Preprocess → ONNX → Embedding → Cosine Similarity → Threshold Decision

## Model Details
- Backbone: MobileNetV2
- Embedding dimension: 512
- Input size: 112×112
- Normalization: ImageNet mean/std
- Decision metric: Cosine similarity

## Running the Project

### Backend (Docker)
```bash
docker build -t face-match-api .
docker run -p 8000:8000 face-match-api
