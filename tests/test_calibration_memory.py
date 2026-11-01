import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.conversation import ConversationMemory


def test_emotion_feedback_adjusts_boost():
    mem = ConversationMemory()
    mem.start_new_session("cal_test")
    assert mem.get_fusion_text_boost() == 0.0
    mem.record_emotion_feedback(False)
    assert mem.get_fusion_text_boost() >= 0.05
    mem.record_emotion_feedback(True)
    assert mem.get_fusion_text_boost() < 0.06
