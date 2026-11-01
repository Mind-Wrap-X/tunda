from src.clinic.profiles import OFFLINE_CARE_MESSAGE, get_clinic_profile
from src.response.generator import EmpathicResponseGenerator, ResponseContext


def _ctx(profile: str) -> ResponseContext:
    return ResponseContext(
        user_text="I had a hard day at work",
        emotion="sad",
        confidence=0.8,
        conversation_history=[],
        empathy_style="supportive",
        user_preferences={"clinic_profile": profile},
        interaction_mode="listen",
    )


def test_care_profiles_disallow_template_fallback():
    between = get_clinic_profile("between_sessions")
    watch = get_clinic_profile("high_risk_watch")
    companion = get_clinic_profile("companion")
    assert between.care_mode and not between.allow_template_fallback
    assert watch.care_mode and not watch.allow_template_fallback
    assert companion.allow_template_fallback and not companion.care_mode
    assert "not a therapist" in between.locked_prompt.lower()


def test_care_mode_offline_does_not_use_templates():
    generator = EmpathicResponseGenerator()
    generator.llm_provider = None
    response = generator.generate_response(_ctx("between_sessions"))
    assert response.care_mode is True
    assert response.llm_online is False
    assert response.interaction_mode == "offline"
    assert "not in care mode" in response.text.lower() or "ollama" in response.text.lower()
    assert response.text == OFFLINE_CARE_MESSAGE


def test_companion_offline_may_use_templates():
    generator = EmpathicResponseGenerator()
    generator.llm_provider = None
    response = generator.generate_response(_ctx("companion"))
    assert response.care_mode is False
    assert response.text != OFFLINE_CARE_MESSAGE
    assert len(response.text) > 10
