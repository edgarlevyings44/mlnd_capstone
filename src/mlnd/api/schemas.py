from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CountyName = Literal["Bungoma", "Kakamega", "Nakuru", "Narok", "Trans Nzoia", "Uasin Gishu"]


def _unit(description: str):
    return Field(..., ge=0, le=1, allow_inf_nan=False, description=description)


class LeafInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
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
            ]
        },
    )

    County: CountyName = Field(..., description="County where the leaf was collected")
    Chlorosis_Intensity: float = _unit("Strength of leaf yellowing (0-1)")
    Green_Yellow_Ratio: float = _unit("Balance of green to yellow area (0-1)")
    Necrotic_Area: float = _unit("Extent of dead tissue (0-1)")
    Texture_Roughness: float = _unit("Roughness of the leaf surface (0-1)")
    Patchiness: float = _unit("Patchy colour pattern (0-1)")
    Lesion_Distribution: float = _unit("Spread of lesions (0-1)")
    Elongated_Streaks: float = _unit("Streak-shaped marks (0-1)")
    Irregular_Lesions: float = _unit("Irregularly shaped lesions (0-1)")
    Infected_Distribution: float = _unit("Spread of the infected area (0-1)")
    Boundary_Sharpness: float = _unit("Sharpness of lesion edges (0-1)")
    Transition_Smoothness: float = _unit("Smoothness of the healthy-to-infected change (0-1)")


class DriverOut(BaseModel):
    feature: str
    label: str
    value: float | str
    contribution: float = Field(description="Contribution to the score in log-odds; contributions add up to the score")
    direction: Literal["raises risk", "lowers risk"]


class PredictResponse(BaseModel):
    label: Literal["flagged", "not_flagged"]
    risk_score: float = Field(description="Screening score between 0 and 1. Not a calibrated probability.")
    threshold: float = Field(description="Scores at or above this are flagged")
    flagged: bool
    band: Literal["clear", "near_threshold", "just_flagged", "flagged", "high_risk"]
    drivers: list[DriverOut]
    model: str
    note: str


class ExplanationOut(BaseModel):
    text: str
    source: Literal["llm", "offline"]
    reason: str | None = Field(default=None, description="Why the offline text was used, when it was")


class ExplainResponse(PredictResponse):
    explanation: ExplanationOut
    recommendations: list[str]
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model: str
    threshold: float
    llm_configured: bool