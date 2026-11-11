from __future__ import annotations

import json
import os
from functools import lru_cache

import anthropic
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query

from mlnd.api.schemas import ExplainResponse, HealthResponse, LeafInput, PredictResponse
from mlnd.config import ROOT, load_config
from mlnd.explain import load_explainer
from mlnd.llm import build_facts, explain_with_llm, recommendations

load_dotenv(ROOT / ".env")

NOTE = "The risk score is a screening score between 0 and 1, not a calibrated probability of infection."
DISCLAIMER = "Decision support only. A laboratory test is needed to confirm MLND."
DESCRIPTION = (
    "Screens maize leaf measurements for Maize Lethal Necrosis Disease (MLND). Returns a risk score, whether the "
    "leaf is flagged, the readings that drove the score (SHAP), and plain-language advice. "
    "The tool supports extension officers; it does not replace laboratory confirmation."
)

app = FastAPI(title="MLND screening API", version="1.0.0", description=DESCRIPTION)


class Service:
    def __init__(self):
        self.cfg = load_config()
        artifacts = ROOT / "artifacts"
        self.explainer = load_explainer(artifacts)
        self.meta = json.loads((artifacts / "model_metadata.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_service() -> Service:
    return Service()


@lru_cache(maxsize=1)
def get_llm_client():
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    lc = load_config()["llm"]
    return anthropic.Anthropic(timeout=lc["timeout_s"], max_retries=lc["max_retries"])


def _score(leaf: LeafInput, svc: Service, k: int) -> tuple[dict, dict, dict]:
    row = pd.DataFrame([leaf.model_dump()])[svc.meta["input_columns"]]
    e = svc.explainer.explain_one(row, k=k)
    rec = recommendations(e["risk_score"], e["threshold"], e["drivers"], svc.cfg)
    body = {
        "label": "flagged" if e["flagged"] else "not_flagged",
        "risk_score": round(e["risk_score"], 4),
        "threshold": e["threshold"],
        "flagged": e["flagged"],
        "band": rec["band"],
        "drivers": e["drivers"],
        "model": svc.meta["model"],
        "note": NOTE,
    }
    return body, rec, e


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health():
    try:
        svc = get_service()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"model not available ({type(exc).__name__})")
    return HealthResponse(
        status="ok",
        model=svc.meta["model"],
        threshold=svc.explainer.threshold,
        llm_configured=bool(os.getenv("ANTHROPIC_API_KEY")),
    )


@app.post("/predict", response_model=PredictResponse, tags=["screening"])
def predict(
    leaf: LeafInput,
    k: int = Query(5, ge=1, le=10, description="Number of top drivers to return"),
    svc: Service = Depends(get_service),
):
    body, _, _ = _score(leaf, svc, k)
    return body


@app.post("/explain", response_model=ExplainResponse, tags=["screening"])
def explain(
    leaf: LeafInput,
    k: int = Query(5, ge=1, le=10, description="Number of top drivers to return"),
    svc: Service = Depends(get_service),
    client=Depends(get_llm_client),
):
    body, rec, e = _score(leaf, svc, k)
    result = explain_with_llm(build_facts(e, leaf.County, k), svc.cfg, client=client)
    return {
        **body,
        "explanation": {"text": result["text"], "source": result["source"], "reason": result.get("reason")},
        "recommendations": rec["actions"],
        "disclaimer": DISCLAIMER,
    }