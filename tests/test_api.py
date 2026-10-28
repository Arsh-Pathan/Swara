"""Integration tests for the FastAPI HTTP endpoints."""

import io
from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from swara.api.app import create_app
from swara.core.config import settings
from swara.engine import Swara


@pytest.fixture
def api_client(temp_dir):
    """Provides a TestClient with a fresh Swara engine instance."""
    test_settings = settings.model_copy()
    test_settings.speaker_store_path = temp_dir / "speakers_api"
    test_settings.speaker_store_path.mkdir(parents=True, exist_ok=True)
    engine = Swara(config=test_settings)

    app = create_app()
    app.state.engine = engine

    with TestClient(app) as client:
        yield client


def _create_wav_bytes(duration_s: float = 1.5, freq: float = 300.0) -> bytes:
    sr = 16000
    t = np.linspace(0, duration_s, int(duration_s * sr), dtype=np.float32)
    # Add harmonics
    sig = 0.4 * np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * freq * 2 * t)
    bio = io.BytesIO()
    sf.write(bio, sig, sr, format="WAV", subtype="PCM_16")
    return bio.getvalue()


def test_api_health(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_api_ready(api_client):
    resp = api_client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ready"] is True
    assert data["models_loaded"] is True


def test_api_models_list(api_client):
    resp = api_client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 2


def test_api_enroll_and_identify(api_client):
    # Use real speech audio for reliable enrollment
    with open("samples/speechbrain_example1.wav", "rb") as f:
        audio_bytes = f.read()

    # Enroll
    enroll_resp = api_client.post(
        "/enroll",
        data={"speaker_id": "api_user", "name": "API User"},
        files={"audio": ("voice.wav", audio_bytes, "audio/wav")},
    )
    assert enroll_resp.status_code == 200
    enroll_data = enroll_resp.json()
    assert enroll_data["speaker_id"] == "api_user"

    # List speakers
    list_resp = api_client.get("/speakers")
    assert list_resp.status_code == 200
    speakers = list_resp.json()
    assert any(s["speaker_id"] == "api_user" for s in speakers)

    # Identify
    ident_resp = api_client.post(
        "/identify",
        files={"audio": ("probe.wav", audio_bytes, "audio/wav")},
    )
    assert ident_resp.status_code == 200
    ident_data = ident_resp.json()
    assert ident_data["is_match"] is True
    assert ident_data["speaker_id"] == "api_user"

    # Verify
    ver_resp = api_client.post(
        "/verify",
        data={"speaker_id": "api_user"},
        files={"audio": ("probe.wav", audio_bytes, "audio/wav")},
    )
    assert ver_resp.status_code == 200
    ver_data = ver_resp.json()
    assert ver_data["verified"] is True

    # Delete
    del_resp = api_client.delete("/speakers/api_user")
    assert del_resp.status_code == 200


def test_api_metrics(api_client):
    resp = api_client.get("/metrics")
    assert resp.status_code == 200
    metrics = resp.json()
    assert "memory_rss_mb" in metrics
    assert "cpu_percent" in metrics
