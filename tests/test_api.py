import json
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlnd.api import main as api_main
from mlnd.api.main import app, get_llm_client, get_service
from mlnd.api.schemas import LeafInput

VALID = {
    "County": "Narok",
    "Chlorosis_Intensity": 0.41,
    "Green_Yellow_Ratio": 0.29,
    "Necrotic_Area": 0.27,
    "Texture_Roughness": 0.60,
    "Patchiness": 0.45,
    "Lesion_Distribution": 0.66,
    "Elongated_Streaks": 1.0,
    "Irregular_Lesions": 0.62,
    "Infected_Distribution": 0.65,
    "Boundary_Sharpness": 0.37,
    "Transition_Smoothness": 0.38,
}


def _with(**changes):
    return {**VALID, **changes}


BAD_PAYLOADS = {
    "feature above 1": _with(Chlorosis_Intensity=1.5),
    "feature below 0": _with(Patchiness=-0.1),
    "unknown county": _with(County="Nairobi"),
    "not a number": _with(Necrotic_Area="high"),
    "missing field": {k: v for k, v in VALID.items() if k != "Texture_Roughness"},
    "extra field": _with(Extra_Field=1),
}


class FakeClient:
    def __init__(self, text):
        self.messages = SimpleNamespace(create=lambda **kw: SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn"))


@pytest.fixture(autouse=True)
def offline_by_default(monkeypatch):
    """Never call the real LLM from tests."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app.dependency_overrides[get_llm_client] = lambda: None
    yield
    app.dependency_overrides.clear()


client = TestClient(app)


def test_health_reports_the_loaded_model():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model"] == "catboost"
    assert body["threshold"] == pytest.approx(get_service().explainer.threshold)


def test_predict_valid_request():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert {"label", "risk_score", "threshold", "flagged", "band", "drivers", "model", "note"} <= set(body)
    assert 0 <= body["risk_score"] <= 1 and len(body["drivers"]) == 5
    assert body["label"] == ("flagged" if body["flagged"] else "not_flagged")
    assert (body["band"] in {"just_flagged", "flagged", "high_risk"}) == body["flagged"]
    assert "not a calibrated probability" in body["note"]


def test_predict_matches_the_model():
    svc = get_service()
    row = pd.DataFrame([VALID])[svc.meta["input_columns"]]
    expected = svc.explainer.pipe.predict_proba(row)[0, 1]
    assert client.post("/predict", json=VALID).json()["risk_score"] == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("endpoint", ["/predict", "/explain"])
@pytest.mark.parametrize("name", list(BAD_PAYLOADS))
def test_bad_requests_return_422(endpoint, name):
    assert client.post(endpoint, json=BAD_PAYLOADS[name]).status_code == 422


def test_explain_falls_back_to_the_offline_text_without_a_key():
    r = client.post("/explain", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["explanation"]["source"] == "offline"
    assert "no API key" in body["explanation"]["reason"]
    assert "risk score" in body["explanation"]["text"]
    assert body["recommendations"] and "laboratory" in body["disclaimer"]


def test_explain_uses_the_llm_when_one_is_available():
    text = "The risk score for this leaf is above the alert threshold, mainly because of the readings listed below."
    app.dependency_overrides[get_llm_client] = lambda: FakeClient(text)
    # Make sure the fake text is consistent with the leaf's actual side of the threshold.
    flagged = client.post("/predict", json=VALID).json()["flagged"]
    if not flagged:
        text = text.replace("above", "below")
        app.dependency_overrides[get_llm_client] = lambda: FakeClient(text)
    body = client.post("/explain", json=VALID).json()
    assert body["explanation"]["source"] == "llm" and body["explanation"]["text"] == text


def test_input_schema_matches_the_trained_model_columns():
    meta = json.loads((api_main.ROOT / "artifacts" / "model_metadata.json").read_text(encoding="utf-8"))
    assert set(LeafInput.model_fields) == set(meta["input_columns"])


def test_swagger_ui_and_openapi_are_served():
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/predict", "/explain"} <= set(paths)


def test_k_controls_the_number_of_drivers():
    assert len(client.post("/predict?k=3", json=VALID).json()["drivers"]) == 3
    assert client.post("/predict?k=0", json=VALID).status_code == 422