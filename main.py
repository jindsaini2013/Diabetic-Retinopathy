from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="DR Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DR_LABELS = {0: "No DR", 1: "Mild DR", 2: "Moderate DR", 3: "Severe DR"}
MODEL_PATH = Path(__file__).resolve().parent / "best_dr_fixed_model.onnx"
INPUT_SIZE = (512, 512)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

session = ort.InferenceSession(
    MODEL_PATH.as_posix(),
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    image_buffer = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unable to decode image.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_channel)
    image = cv2.cvtColor(
        cv2.merge([l_channel, a_channel, b_channel]),
        cv2.COLOR_LAB2RGB,
    )

    image = Image.fromarray(image).resize(INPUT_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = (image_array - MEAN) / STD
    image_array = np.transpose(image_array, (2, 0, 1))
    return np.expand_dims(image_array, axis=0).astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def predict_logits(image_tensor: np.ndarray) -> np.ndarray:
    return session.run(None, {input_name: image_tensor})[0]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    try:
        contents = await file.read()
        image_tensor = preprocess_image(contents)
        logits = predict_logits(image_tensor)
        probabilities = softmax(logits)
        grade = int(np.argmax(probabilities, axis=1)[0])
        confidence = float(probabilities[0][grade])

        return {
            "grade": grade,
            "label": DR_LABELS[grade],
            "confidence": confidence,
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {str(exc)}") from exc


@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    raise HTTPException(
        501,
        "Grad-CAM is disabled in the Vercel deployment because the API now runs on ONNX Runtime instead of PyTorch.",
    )
