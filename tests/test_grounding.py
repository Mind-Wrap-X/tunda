from src.response.generator import EmpathicResponseGenerator, ResponseContext
from src.response.grounding import select_grounding, should_offer_grounding


def test_explicit_request_offers_grounding():
    assert should_offer_grounding("Help me ground, I can't calm down", "anxious", "listen")


def test_panic_uses_box_breath():
    script = select_grounding("I'm having a panic attack and I can't breathe", "anxious", "listen")
    assert script is not None
    assert script.script_id == "box"
    assert any(step.pause_ms >= 4000 for step in script.steps)


def test_listen_mode_without_distress_skips_grounding():
    assert select_grounding("I had lunch with a friend", "happy", "listen") is None


def test_coach_anxious_offers_paced_steps():
    script = select_grounding("I don't know what to do", "anxious", "coach")
    assert script is not None
    assert len(script.steps) >= 4
    payload = script.to_payload()
    assert payload["steps"][0]["text"]


def test_crisis_mode_does_not_replace_safety():
    assert should_offer_grounding("help me ground", "sad", "escalate") is False


def test_generator_returns_grounding_payload():
    generator = EmpathicResponseGenerator()
    context = ResponseContext(
        user_text="Please guide me through a box breathing exercise",
        emotion="anxious",
        confidence=0.9,
        conversation_history=[],
        empathy_style="supportive",
        user_preferences={},
        interaction_mode="listen",
    )
    response = generator.generate_response(context)
    assert response.grounding is not None
    assert response.grounding["id"] == "box"
    assert "Breathe in" in response.text
