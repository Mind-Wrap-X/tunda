from src.response.safety import SafetyGuard, list_crisis_regions


def test_self_harm_detection():
    guard = SafetyGuard()
    result = guard.assess("I want to kill myself", session_id="s1")
    assert result.is_crisis is True
    assert result.tier == "self_harm"
    assert result.response is not None
    assert "988" in result.response


def test_distress_does_not_block_first_time():
    guard = SafetyGuard()
    result = guard.assess("I feel hopeless and worthless", session_id="d1")
    assert result.is_crisis is False
    assert result.tier == "distress"
    assert result.support_note is None


def test_repeated_distress_adds_local_help():
    guard = SafetyGuard()
    guard.assess("I feel hopeless", session_id="d2")
    result = guard.assess("I still feel hopeless", session_id="d2")
    assert result.is_crisis is False
    assert result.repeated is True
    assert result.support_note is not None
    assert "988" in result.support_note


def test_imminent_uses_emergency_number():
    guard = SafetyGuard()
    result = guard.assess("I have a plan to kill myself tonight", session_id="i1")
    assert result.tier == "imminent"
    assert result.is_crisis is True
    assert "911" in result.response


def test_uk_region_uses_local_numbers():
    guard = SafetyGuard()
    guard.region = "UK"
    result = guard.assess("I want to kill myself", session_id="uk1")
    assert "999" in result.response
    assert "116 123" in result.response


def test_generator_honors_region_preference():
    from src.response.generator import EmpathicResponseGenerator, ResponseContext

    generator = EmpathicResponseGenerator()
    context = ResponseContext(
        user_text="I want to kill myself",
        emotion="sad",
        confidence=0.9,
        conversation_history=[],
        empathy_style="supportive",
        user_preferences={"session_id": "uk-ui", "region": "UK"},
    )
    response = generator.generate_response(context)
    assert "999" in response.text
    assert response.crisis_resources["crisis_line"] == "116 123"


def test_list_crisis_regions_includes_local_numbers():
    regions = {row["code"]: row for row in list_crisis_regions()}
    assert regions["US"]["crisis_line"] == "988"
    assert regions["UK"]["emergency"] == "999"
    assert "INTL" in regions


def test_repeated_self_harm_escalates():
    guard = SafetyGuard()
    first = guard.assess("I want to die", session_id="r1")
    second = guard.assess("I want to kill myself", session_id="r1")
    assert first.tier == "self_harm"
    assert second.tier == "imminent"
    assert "911" in second.response
