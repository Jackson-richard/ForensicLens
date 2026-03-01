import pytest
from fastapi.testclient import TestClient
from backend.main import app
import io
from PIL import Image

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_analyze_image():
    
    img = Image.new('RGB', (224, 224), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    response = client.post(
        "/api/analyze/image",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "authenticity" in data
    assert "synthetic_type" in data
    assert "confidence" in data
    assert "attribution_family" in data
    assert "pdf_report" in data
    assert "heatmap" in data
    assert "forensic_indicators" in data
    
    indicators = data["forensic_indicators"]
    assert "metadata_present" in indicators
    assert "high_frequency_anomaly" in indicators
    assert "noise_inconsistency" in indicators
    assert "face_distortion_detected" in indicators
