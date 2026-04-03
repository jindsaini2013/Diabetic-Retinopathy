import io
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        return x * self.spatial_att(x)


class LightCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.attention = CBAMBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = nn.Sequential()

        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.residual(x)
        out = self.bottleneck(x)
        out = self.conv(out)
        out = self.attention(out)
        out += identity
        return self.relu(out)


class EfficientDRNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 3)
        self.layer3 = self._make_layer(256, 256, 3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def _make_layer(self, in_ch, out_ch, blocks):
        layers = [LightCNNBlock(in_ch, out_ch), nn.MaxPool2d(2)]
        for _ in range(blocks - 1):
            layers.append(LightCNNBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


app = FastAPI(title="DR Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cpu")
DR_LABELS = {0: "No DR", 1: "Mild DR", 2: "Moderate DR", 3: "Severe DR"}
MODEL_PATH = Path(__file__).resolve().parent / "best_dr_fixed_model.pth"

model = EfficientDRNet(num_classes=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    image_buffer = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unable to decode image.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_channel)
    image = cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2RGB)
    pil_image = Image.fromarray(image)
    return transform(pil_image).unsqueeze(0).to(DEVICE)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    try:
        contents = await file.read()
        tensor = preprocess_image(contents)
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            grade = int(probabilities.argmax(1).item())
            confidence = float(probabilities[0][grade].item())
            class_probabilities = {
                str(index): {
                    "label": DR_LABELS[index],
                    "probability": float(probabilities[0][index].item()),
                    "percentage": round(float(probabilities[0][index].item()) * 100, 2),
                }
                for index in sorted(DR_LABELS)
            }
        return {
            "grade": grade,
            "label": DR_LABELS[grade],
            "confidence": confidence,
            "class_probabilities": class_probabilities,
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {str(exc)}") from exc


@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")

    try:
        contents = await file.read()
        tensor = preprocess_image(contents)
        tensor.requires_grad_(True)

        gradients = []
        activations = []

        def save_grad(_, __, grad_output):
            gradients.append(grad_output[0])

        def save_activation(_, __, output):
            activations.append(output)

        target_layer = model.layer3[-1].conv[-2]
        forward_hook = target_layer.register_forward_hook(save_activation)
        backward_hook = target_layer.register_backward_hook(save_grad)

        outputs = model(tensor)
        predicted_class = outputs.argmax(1).item()
        model.zero_grad()
        outputs[0, predicted_class].backward()

        forward_hook.remove()
        backward_hook.remove()

        grads = gradients[0].detach().cpu().numpy()[0]
        acts = activations[0].detach().cpu().numpy()[0]

        weights = grads.mean(axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for index, weight in enumerate(weights):
            cam += weight * acts[index]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (512, 512))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        original = np.frombuffer(contents, np.uint8)
        original = cv2.imdecode(original, cv2.IMREAD_COLOR)
        original = cv2.resize(original, (512, 512))
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        _, buffer = cv2.imencode(".jpg", overlay)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Grad-CAM failed: {str(exc)}") from exc
