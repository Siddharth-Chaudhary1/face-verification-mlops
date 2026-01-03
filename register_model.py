import mlflow
import mlflow.onnx
import onnx

MODEL_PATH = "model/facedet.onnx"
THRESHOLD = 0.35

mlflow.set_experiment("face-verification")

with mlflow.start_run():
    mlflow.log_param("model_type", "mobilenetv2")
    mlflow.log_param("embedding_dim", 512)
    mlflow.log_param("input_size", "112x112")
    mlflow.log_param("threshold", THRESHOLD)

    onnx_model = onnx.load(MODEL_PATH)
    mlflow.onnx.log_model(onnx_model, artifact_path="model")

print("✅ Model registered in MLflow")
