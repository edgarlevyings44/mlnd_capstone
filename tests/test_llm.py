import json
from types import SimpleNamespace

from mlnd import llm
from mlnd.config import load_config

CFG = load_config()
THRESHOLD = 0.4
FACTS = {
    "county": "Narok",
    "risk_score": 0.92,
    "alert_threshold": 0.4,
    "above_threshold": True,
    "drivers": [
        {"reading": "Streak-shaped marks", "value": 1.0, "effect": "raises risk"},
        {"reading": "Spread of the infected area", "value": 0.098, "effect": "lowers risk"},
    ],
}
GOOD = (
    "The tool gives this leaf a risk score of 0.92, above the alert threshold of 0.40. "
    "Streak-shaped marks at 1.0 raised the score most, while the spread of the infected area at 0.098 lowered it."
)


class FakeClient:
    def __init__(self, text=GOOD, stop_reason="end_turn", error=None):
        self.calls = []
        self._text, self._stop, self._error = text, stop_reason, error
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._text)], stop_reason=self._stop)


def test_prompt_contains_only_the_supplied_facts():
    system, user = llm.build_messages(FACTS)
    block = user.split("FACTS (JSON):\n", 1)[1].split("\n\nWrite", 1)[0]
    assert json.loads(block) == FACTS
    for topic in ("pesticide", "climate", "doses", "probability"):
        assert topic in system.lower()
    assert "most to least influential" in system
    assert "'all'" in system


def test_success_uses_the_configured_model_and_sends_no_sampling_parameters():
    client = FakeClient()
    out = llm.explain_with_llm(FACTS, CFG, client=client)
    assert out["source"] == "llm" and out["text"] == GOOD
    sent = client.calls[0]
    assert sent["model"] == CFG["llm"]["model"] and sent["messages"][0]["role"] == "user"
    assert not {"temperature", "top_p", "top_k"} & set(sent)


def test_guardrail_replaces_invented_content_with_the_offline_text():
    bad = "Spray cypermethrin at 400 ml per 20 litres because the humid altitude favours thrips."
    out = llm.explain_with_llm(FACTS, CFG, client=FakeClient(text=bad))
    assert out["source"] == "offline" and "guardrail" in out["reason"]
    assert llm.validate_explanation(out["text"], FACTS) == []


def test_sweeping_claims_fall_back_to_the_offline_text():
    bad = "All measured leaf features lowered the risk score."
    out = llm.explain_with_llm(FACTS, CFG, client=FakeClient(text=bad))
    assert out["source"] == "offline" and "guardrail" in out["reason"]


def test_llm_errors_fall_back_instead_of_raising():
    out = llm.explain_with_llm(FACTS, CFG, client=FakeClient(error=RuntimeError("boom")))
    assert out["source"] == "offline" and "RuntimeError" in out["reason"]


def test_missing_api_key_falls_back(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = llm.explain_with_llm(FACTS, CFG)
    assert out["source"] == "offline" and "no API key" in out["reason"]


def test_truncated_or_refused_answers_fall_back():
    for stop in ("max_tokens", "refusal"):
        out = llm.explain_with_llm(FACTS, CFG, client=FakeClient(stop_reason=stop))
        assert out["source"] == "offline" and stop in out["reason"]


def test_validator_accepts_grounded_text_and_rejects_the_rest():
    assert llm.validate_explanation(GOOD, FACTS) == []
    assert llm.validate_explanation("The risk is 92% likely.", FACTS)
    assert llm.validate_explanation("There is a high probability of infection.", FACTS)
    assert llm.validate_explanation("The score was 0.77.", FACTS)
    assert llm.validate_explanation("Humid weather raised the score.", FACTS)
    assert llm.validate_explanation("All measured leaf features lowered the risk score.", FACTS)
    assert llm.validate_explanation("Every reading lowered the score.", FACTS)


def test_offline_explanation_is_informative_and_clean():
    text = llm.offline_explanation(FACTS)
    assert "0.92" in text and "above" in text and "streak-shaped marks" in text
    assert llm.validate_explanation(text, FACTS) == []


def test_bands_follow_the_score_and_threshold():
    lc = CFG["llm"]

    def pick(r):
        return llm.risk_band(r, THRESHOLD, lc["band_margin"], lc["high_risk_at"])

    assert pick(0.10) == "clear"
    assert pick(0.30) == "near_threshold"
    assert pick(0.45) == "just_flagged"
    assert pick(0.60) == "flagged"
    assert pick(0.90) == "high_risk"
    assert pick(0.40) == "just_flagged" and pick(0.399) == "near_threshold"


def test_every_band_has_clean_text_and_only_flagged_bands_mention_a_laboratory():
    for actions in llm.ACTIONS.values():
        assert actions and llm.forbidden_terms(" ".join(actions)) == []
    assert not any("laboratory" in a for a in llm.ACTIONS["clear"] + llm.ACTIONS["near_threshold"])
    for band in ("just_flagged", "flagged", "high_risk"):
        assert any("laboratory" in a for a in llm.ACTIONS[band])


def test_look_for_hint_uses_only_the_readings_that_raised_the_score():
    drivers = [
        {"feature": "Elongated_Streaks", "direction": "raises risk"},
        {"feature": "Patchiness", "direction": "lowers risk"},
        {"feature": "Irregular_Lesions", "direction": "raises risk"},
    ]
    out = llm.recommendations(0.92, THRESHOLD, drivers, CFG)
    text = " ".join(out["actions"])
    assert out["band"] == "high_risk"
    assert "streak-shaped marks" in text and "irregularly shaped lesions" in text
    assert "patchy" not in text
    clear = llm.recommendations(0.10, THRESHOLD, drivers, CFG)
    assert clear["band"] == "clear" and "look closely" not in " ".join(clear["actions"])


def test_recommendation_text_is_clean_in_every_band():
    drivers = [{"feature": f, "direction": "raises risk"} for f in list(llm.LOOK_FOR)[:3]]
    for score in (0.10, 0.30, 0.45, 0.60, 0.90):
        out = llm.recommendations(score, THRESHOLD, drivers, CFG)
        assert llm.forbidden_terms(" ".join(out["actions"])) == []