from __future__ import annotations

import argparse
import json
import os
import re

import anthropic
from dotenv import load_dotenv

from mlnd.config import ROOT, load_config

SYSTEM_PROMPT = """You help explain the output of a maize leaf screening tool to farmers and agricultural extension officers in Kenya. The tool gives a risk score (0 to 1) that a leaf is infected with Maize Lethal Necrosis Disease (MLND), computed from numeric leaf measurements.

Write 2 to 3 plain sentences explaining why the tool gave this score, using ONLY the facts in the FACTS block. Rules:
- Mention the risk score and whether it is above the alert threshold.
- Name the most influential readings and say whether it raised or lowered the score.
- The readings in FACTS are listed from most to least influential. Describe them in that order, and never say that a later reading was more influential than an earlier one.
- Describe only the readings listed in FACTS. Never use the words 'all', 'every' or 'each' about readings, features or measurements; say 'the main readings' instead.
- Use only numbers that appear in FACTS.
- Call it a risk score. Never call it a probability and never use percentages.
- Do not mention weather, climate, altitude, soil, insects or vectors, pesticides, chemicals, doses, or any fact that is not in FACTS.
- Do not give advice or recommendations; those are added separately.
- Do not claim a diagnosis. This is a screening aid, and only a laboratory test confirms MLND.
- Plain text only, under 80 words."""

FORBIDDEN = {
    "climate or environment": re.compile(
        r"\b(rain\w*|humid\w*|temperature\w*|altitude|elevation|climat\w*|weather|season\w*|drought|soil|wind|moist\w*)\b", re.I),
    "insects or vectors": re.compile(
        r"\b(thrips|beetle\w*|aphid\w*|leafhopper\w*|insect\w*|vector\w*|pest\w*|mite\w*)\b", re.I),
    "chemicals or doses": re.compile(
        r"\b(pesticide\w*|insecticide\w*|fungicide\w*|herbicide\w*|spray\w*|chemical\w*|cypermethrin|dose\w*|dosage|litre\w*|liter\w*|mg|ml|kg)\b", re.I),
}
NUMBER = re.compile(r"\d+(?:\.\d+)?")
SWEEPING = re.compile(
    r"\b(all|every|each)\b[^.]{0,40}\b(features?|readings?|measurements?|factors?|signs?|symptoms?)\b", re.I)

ACTIONS = {
    "clear": [
        "No action is needed from this reading alone; keep up routine scouting.",
        "Re-check the plant if symptoms change or spread.",
    ],
    "near_threshold": [
        "The score is close to the alert threshold, so this plant is not flagged but is worth watching.",
        "Re-check this plant in a few days, and sooner if symptoms spread.",
    ],
    "just_flagged": [
        "The score is only just above the alert threshold.",
        "Re-inspect this plant in a few days and send a sample to a laboratory if symptoms persist or worsen.",
        "Check neighbouring plants for similar symptoms.",
    ],
    "flagged": [
        "Ask an extension officer to inspect this plant and confirm it with a laboratory test.",
        "Check neighbouring plants for similar symptoms.",
        "Avoid moving plant material out of the field until the result is confirmed.",
    ],
    "high_risk": [
        "Prioritise a laboratory sample for this plant.",
        "Ask an extension officer to inspect it as soon as possible.",
        "Check neighbouring plants for similar symptoms.",
        "Avoid moving plant material out of the field until the result is confirmed.",
    ],
}

LOOK_FOR = {
    "Chlorosis_Intensity": "yellowing of the leaf",
    "Green_Yellow_Ratio": "yellowing of the leaf",
    "Necrotic_Area": "dead (brown) tissue",
    "Texture_Roughness": "a rough leaf surface",
    "Patchiness": "patchy colour",
    "Lesion_Distribution": "lesions spreading across the leaf",
    "Elongated_Streaks": "streak-shaped marks",
    "Irregular_Lesions": "irregularly shaped lesions",
    "Infected_Distribution": "the spread of the damaged area",
    "Boundary_Sharpness": "sharp edges around lesions",
    "Transition_Smoothness": "how abruptly healthy tissue turns damaged",
    "Severity_Score": "overall leaf damage",
    "Chlorotic_Necrotic_Index": "yellowing together with dead tissue",
    "Lesion_Severity_Ratio": "streaks and irregular lesions",
    "Necrotic_Texture_Interaction": "dead tissue on a rough surface",
    "Boundary_Transition_Diff": "how sharply lesions end",
}


def risk_band(risk_score: float, threshold: float, margin: float, high_risk_at: float) -> str:
    if risk_score >= high_risk_at:
        return "high_risk"
    if risk_score >= threshold + margin:
        return "flagged"
    if risk_score >= threshold:
        return "just_flagged"
    if risk_score >= threshold - margin:
        return "near_threshold"
    return "clear"


def recommendations(risk_score: float, threshold: float, drivers=None, cfg: dict | None = None) -> dict:
    lc = (cfg or load_config())["llm"]
    band = risk_band(risk_score, threshold, lc["band_margin"], lc["high_risk_at"])
    actions = list(ACTIONS[band])
    if band != "clear":
        phrases: list[str] = []
        for d in drivers or []:
            phrase = LOOK_FOR.get(d.get("feature"))
            if d.get("direction") == "raises risk" and phrase and phrase not in phrases:
                phrases.append(phrase)
            if len(phrases) == 2:
                break
        if phrases:
            actions.insert(1, "When you inspect it, look closely for: " + " and ".join(phrases) + ".")
    return {"band": band, "actions": actions}


def build_facts(explanation: dict, county: str, k: int = 5) -> dict:
    return {
        "county": county,
        "risk_score": round(explanation["risk_score"], 2),
        "alert_threshold": round(explanation["threshold"], 2),
        "above_threshold": bool(explanation["flagged"]),
        "drivers": [
            {"reading": d["label"], "value": d["value"], "effect": d["direction"]}
            for d in explanation["drivers"][:k]
        ],
    }


def build_messages(facts: dict) -> tuple[str, str]:
    user = "FACTS (JSON):\n" + json.dumps(facts, indent=2) + "\n\nWrite the explanation."
    return SYSTEM_PROMPT, user


def forbidden_terms(text: str) -> list[str]:
    found = []
    for name, rx in FORBIDDEN.items():
        m = rx.search(text)
        if m:
            found.append(f"{name}: '{m.group(0)}'")
    return found


def allowed_numbers(facts: dict) -> set[str]:
    values = [facts["risk_score"], facts["alert_threshold"]]
    values += [d["value"] for d in facts["drivers"] if isinstance(d["value"], (int, float))]
    allowed = {"0", "1"}
    for v in values:
        for k in range(4):
            allowed.add(f"{round(float(v), k):g}")
            allowed.add(f"{float(v):.{k}f}")
    return allowed


def validate_explanation(text: str, facts: dict) -> list[str]:
    problems = []
    t = text.strip()
    if not t:
        problems.append("empty answer")
    if "%" in t:
        problems.append("uses a percentage")
    if re.search(r"\bprobabilit", t, re.I):
        problems.append("calls the score a probability")
    problems += forbidden_terms(t)
    m = SWEEPING.search(t)
    if m:
        problems.append(f"sweeping claim about the readings: '{m.group(0)}'")
    ok = allowed_numbers(facts)
    problems += [f"number not in the facts: {n}" for n in NUMBER.findall(t) if n not in ok]
    if len(t.split()) > 140:
        problems.append("too long")
    return problems


def offline_explanation(facts: dict) -> str:
    status = "above" if facts["above_threshold"] else "below"
    up = [d for d in facts["drivers"] if d["effect"] == "raises risk"]
    down = [d for d in facts["drivers"] if d["effect"] == "lowers risk"]

    def fmt(items):
        return "; ".join(f"{d['reading'].lower()} ({d['value']})" for d in items)

    parts = [f"The tool gives this leaf a risk score of {facts['risk_score']:.2f}, {status} the alert threshold of {facts['alert_threshold']:.2f}."]
    if up:
        parts.append(f"Readings that raised the score most: {fmt(up)}.")
    if down:
        parts.append(f"Readings that lowered it: {fmt(down)}.")
    parts.append("This is a screening aid, not a diagnosis.")
    return " ".join(parts)


def _offline(facts: dict, reason: str) -> dict:
    return {"text": offline_explanation(facts), "source": "offline", "reason": reason}


def explain_with_llm(facts: dict, cfg: dict | None = None, client=None) -> dict:
    cfg = cfg or load_config()
    lc = cfg["llm"]
    if client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            return _offline(facts, "no API key configured")
        client = anthropic.Anthropic(timeout=lc["timeout_s"], max_retries=lc["max_retries"])

    system, user = build_messages(facts)
    kwargs = {
        "model": lc["model"],
        "max_tokens": lc["max_tokens"],
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if lc.get("effort"):
        kwargs["output_config"] = {"effort": lc["effort"]}

    try:
        response = client.messages.create(**kwargs)
    except Exception as e:
        return _offline(facts, f"LLM call failed ({type(e).__name__})")

    if response.stop_reason != "end_turn":
        return _offline(facts, f"LLM stopped with '{response.stop_reason}'")
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    problems = validate_explanation(text, facts)
    if problems:
        return _offline(facts, "guardrail rejected the answer: " + "; ".join(problems))
    return {"text": text, "source": "llm", "model": lc["model"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Show grounded explanations for the saved example leaves.")
    ap.add_argument("--offline", action="store_true", help="ignore the API key and use the offline text")
    args = ap.parse_args(argv)

    load_dotenv(ROOT / ".env")
    if args.offline:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    cfg = load_config()
    path = ROOT / cfg["paths"]["reports_dir"] / "shap_example_explanations.json"
    examples = json.loads(path.read_text(encoding="utf-8"))
    for name, e in examples.items():
        facts = build_facts(e, e["input"]["County"], cfg["llm"]["top_k_drivers"])
        result = explain_with_llm(facts, cfg)
        print(f"\n== {name} ==")
        print(f"source: {result['source']}" + (f" ({result['reason']})" if "reason" in result else ""))
        print(result["text"])
        rec = recommendations(e["risk_score"], e["threshold"], e["drivers"], cfg)
        print(f"Recommendations ({rec['band']}):")
        for r in rec["actions"]:
            print(f"  - {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())