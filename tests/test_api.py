import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_grade():
    img = Image.new("RGB", (512, 512), color=(200, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.png", buf, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "grade" in data
    assert "label" in data
    assert "confidence" in data
    assert "class_probabilities" in data
    assert len(data["class_probabilities"]) == 4
    assert sum(item["probability"] for item in data["class_probabilities"].values()) == pytest.approx(1.0, rel=1e-5)
    assert data["grade"] in (0, 1, 2, 3)


def test_predict_rejects_non_image():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400
