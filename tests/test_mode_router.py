import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.response.mode_router import InteractionModeRouter, mode_instructions


def test_default_listen():
    r = InteractionModeRouter()
    d = r.route("I had a rough day at work", {})
    assert d.mode == "listen"


def test_explicit_coach():
    r = InteractionModeRouter()
    d = r.route("Help me calm down, what should I do?", {})
    assert d.mode == "coach"


def test_explicit_listen_override():
    r = InteractionModeRouter()
    d = r.route("Please just listen, no advice", {})
    assert d.mode == "listen"


def test_user_preference_coach():
    r = InteractionModeRouter()
    d = r.route("anything", {"response_mode": "coach"})
    assert d.mode == "coach"


def test_mode_instructions_nonempty():
    assert "LISTEN" in mode_instructions("listen")
    assert "COACH" in mode_instructions("coach")
