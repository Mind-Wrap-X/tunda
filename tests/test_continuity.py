from datetime import datetime, timedelta

from cryptography.fernet import Fernet

from src.memory.continuity import build_shareable_recap, clinician_trends
from src.memory.conversation import ConversationMemory, ConversationSession, ConversationTurn


def _turn(text, emotion="anxious", when=None):
    return ConversationTurn(
        timestamp=(when or datetime.now()).isoformat(),
        user_text=text,
        user_emotion=emotion,
        user_confidence=0.8,
        assistant_response="I hear you.",
        empathy_style="supportive",
        response_confidence=0.7,
    )


def test_shareable_recap_is_five_lines_without_quotes():
    start = datetime.now() - timedelta(minutes=12)
    session = ConversationSession(
        session_id="s1",
        start_time=start.isoformat(),
        end_time=datetime.now().isoformat(),
        turns=[
            _turn("I cannot sleep and work is crushing me", "anxious", start),
            _turn("I still feel panicked about my job", "anxious"),
        ],
        emotion_history=[],
        user_preferences={},
        session_summary="",
        persist_consent=True,
        safety_flags=["distress"],
        tools_used=["grounding"],
    )
    recap = build_shareable_recap(session)
    lines = recap.split("\n")
    assert len(lines) == 5
    assert "crushing me" not in recap
    assert "sleep" in recap.lower() or "work" in recap.lower()
    assert "distress" in recap


def test_clinician_trends_omit_transcripts():
    session = ConversationSession(
        session_id="s2",
        start_time=datetime.now().isoformat(),
        end_time=datetime.now().isoformat(),
        turns=[_turn("secret private sentence about my sister")],
        emotion_history=[],
        user_preferences={},
        session_summary="",
        persist_consent=True,
        clinician_share=True,
        safety_flags=["self_harm"],
    )
    payload = clinician_trends([session])
    assert payload["includes_transcripts"] is False
    blob = str(payload)
    assert "sister" not in blob
    assert payload["days"][0]["crisis_flags"]["self_harm"] == 1


def test_resume_and_recap_on_memory(tmp_path):
    mem = ConversationMemory(
        conversation_file=str(tmp_path / "c.json"),
        memory_key=Fernet.generate_key(),
    )
    mem.start_new_session("old")
    mem.add_conversation_turn(
        user_text="I feel hopeless at work",
        user_emotion="sad",
        user_confidence=0.9,
        assistant_response="I'm with you.",
        empathy_style="supportive",
        response_confidence=0.8,
    )
    mem.set_persist_consent(True)
    mem.end_current_session()
    mem.start_new_session("new")
    preview = mem.last_persisted_preview()
    assert preview and "Date:" in preview["recap"]
    result = mem.resume_last_session()
    assert result["resumed"] is True
    assert mem.current_session.user_preferences.get("resume_recap")
