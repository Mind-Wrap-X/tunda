"""Shareable recaps and clinician trends — no transcript dump."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
import re

_THEME_BUCKETS = {
    "sleep": ("sleep", "insomnia", "tired", "exhausted", "nightmare"),
    "work": ("work", "job", "boss", "school", "exam"),
    "family": ("family", "mom", "dad", "partner", "kids", "child"),
    "grief": ("grief", "loss", "died", "funeral", "missing"),
    "panic": ("panic", "anxious", "anxiety", "overwhelm", "racing"),
    "anger": ("angry", "rage", "furious", "irritated"),
    "loneliness": ("alone", "lonely", "isolated"),
    "hope": ("hope", "better", "grateful", "relief"),
}


def _parse_when(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _day_key(value: Optional[str]) -> str:
    parsed = _parse_when(value)
    return parsed.date().isoformat() if parsed else "unknown"


def extract_themes(texts: Iterable[str], limit: int = 3) -> List[str]:
    found: List[str] = []
    blob = " ".join(texts).lower()
    for label, words in _THEME_BUCKETS.items():
        if any(re.search(rf"\b{re.escape(word)}\b", blob) for word in words):
            found.append(label)
        if len(found) >= limit:
            break
    return found


def build_shareable_recap(session: Any) -> str:
    """Five short lines a patient can copy. No quoted utterances."""
    turns = getattr(session, "turns", []) or []
    emotions = [getattr(t, "user_emotion", "neutral") for t in turns]
    start = _parse_when(getattr(session, "start_time", None))
    end = _parse_when(getattr(session, "end_time", None)) or (
        _parse_when(getattr(turns[-1], "timestamp", None)) if turns else None
    )
    minutes = 0
    if start and end:
        minutes = max(1, int((end - start).total_seconds() / 60))
    mood = Counter(emotions).most_common(1)[0][0] if emotions else "not labeled"
    flags = list(getattr(session, "safety_flags", []) or [])
    tools = list(getattr(session, "tools_used", []) or [])
    themes = extract_themes(getattr(t, "user_text", "") for t in turns)
    date_label = start.strftime("%b %d, %Y") if start else "Today"

    line1 = f"Date: {date_label}. About {minutes} minute(s), {len(turns)} exchange(s)."
    line2 = f"Mood: mostly {mood}."
    line3 = (
        f"Themes: {', '.join(themes)}."
        if themes
        else "Themes: general check-in (no specific topic tagged)."
    )
    if flags:
        unique = ", ".join(sorted(set(flags)))
        line4 = f"Safety: {unique} flagged. No message text is included here."
    else:
        line4 = "Safety: no crisis flags recorded."
    if "grounding" in tools:
        line5 = "Next: a grounding exercise was used. You can pick up here or start fresh."
    else:
        line5 = "Next: you can pick up from here or start a new visit."
    return "\n".join([line1, line2, line3, line4, line5])


def clinician_trends(sessions: List[Any]) -> Dict[str, Any]:
    """Aggregates only: mood, crisis flags, engagement. Never transcripts."""
    by_day: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "sessions": 0,
            "turns": 0,
            "minutes": 0,
            "emotions": Counter(),
            "crisis_flags": Counter(),
        }
    )
    for session in sessions:
        if getattr(session, "persist_consent", None) is not True:
            continue
        day = _day_key(getattr(session, "start_time", None))
        bucket = by_day[day]
        bucket["sessions"] += 1
        turns = getattr(session, "turns", []) or []
        bucket["turns"] += len(turns)
        start = _parse_when(getattr(session, "start_time", None))
        end = _parse_when(getattr(session, "end_time", None))
        if start and end:
            bucket["minutes"] += max(1, int((end - start).total_seconds() / 60))
        for turn in turns:
            bucket["emotions"][getattr(turn, "user_emotion", "neutral")] += 1
        for flag in getattr(session, "safety_flags", []) or []:
            bucket["crisis_flags"][flag] += 1

    days = []
    for day in sorted(by_day):
        row = by_day[day]
        top_mood = row["emotions"].most_common(1)[0][0] if row["emotions"] else "none"
        days.append(
            {
                "date": day,
                "sessions": row["sessions"],
                "turns": row["turns"],
                "minutes": row["minutes"],
                "top_mood": top_mood,
                "mood_counts": dict(row["emotions"]),
                "crisis_flags": dict(row["crisis_flags"]),
            }
        )
    return {
        "includes_transcripts": False,
        "session_count": sum(1 for s in sessions if getattr(s, "persist_consent", None) is True),
        "days": days,
    }
