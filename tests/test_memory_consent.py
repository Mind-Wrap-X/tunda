"""Consent-gated, encrypted conversation storage."""

import json
from pathlib import Path

from cryptography.fernet import Fernet

from src.memory.conversation import ConversationMemory


def _memory(tmp_path: Path) -> ConversationMemory:
    key = Fernet.generate_key()
    return ConversationMemory(
        conversation_file=str(tmp_path / "conversations.json"),
        memory_key=key,
    )


def _add_turn(mem: ConversationMemory, text: str = "hello") -> None:
    mem.add_conversation_turn(
        user_text=text,
        user_emotion="neutral",
        user_confidence=0.8,
        assistant_response="I hear you.",
        empathy_style="supportive",
        response_confidence=0.7,
    )


def test_no_consent_does_not_write_file(tmp_path):
    mem = _memory(tmp_path)
    mem.start_new_session("s_private")
    _add_turn(mem)
    mem.end_current_session()
    assert not Path(mem.conversation_file).exists() or Path(mem.conversation_file).read_text().strip() in {"", "[]"}


def test_remember_writes_encrypted_envelope(tmp_path):
    mem = _memory(tmp_path)
    mem.start_new_session("s_keep")
    _add_turn(mem)
    status = mem.set_persist_consent(True)
    assert status["persist_consent"] is True
    assert status["encrypted"] is True
    raw = Path(mem.conversation_file).read_text(encoding="utf-8")
    envelope = json.loads(raw)
    assert envelope["tunda_memory"] == 1
    assert "hello" not in raw
    assert "I hear you" not in raw


def test_forget_removes_stored_session(tmp_path):
    mem = _memory(tmp_path)
    mem.start_new_session("s_forget")
    _add_turn(mem)
    mem.set_persist_consent(True)
    mem.set_persist_consent(False)
    data = mem._store.read()
    assert data == []


def test_spoken_remember_this_enables_persist(tmp_path):
    mem = _memory(tmp_path)
    mem.start_new_session("s_speak")
    _add_turn(mem, "Please remember this conversation")
    assert mem.current_session.persist_consent is True
    raw = Path(mem.conversation_file).read_text(encoding="utf-8")
    assert json.loads(raw)["alg"] == "fernet"
