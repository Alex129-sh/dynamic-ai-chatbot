"""Lightweight NLP: sentiment (VADER), intent classification, simple NER."""

from __future__ import annotations

import re
from typing import Any

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _analyzer = SentimentIntensityAnalyzer()
except Exception:  # pragma: no cover
    _analyzer = None


# Simple keyword-based intent patterns (order matters — first match wins)
_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("greeting", [r"\b(hi|hello|hey|namaste|good morning|good evening|good afternoon)\b"]),
    ("farewell", [r"\b(bye|goodbye|see you|take care|good night)\b"]),
    ("thanks", [r"\b(thanks|thank you|thx|appreciate)\b"]),
    ("identity", [r"\b(who are you|what are you|your name|about yourself)\b"]),
    ("help", [r"\b(help|how (can|do) i|what can you do|assist)\b"]),
    ("weather", [r"\b(weather|temperature|rain|forecast)\b"]),
    ("code", [r"\b(code|python|javascript|function|bug|error|debug|programming)\b"]),
    ("math", [r"\b(calculate|compute|solve|equation|math|sum|average|mean|median)\b"]),
    ("document_qa", [r"\b(pdf|document|file|sheet|slide|page|according to|from the)\b"]),
    ("memory", [r"\b(remember|yaad rakhna|forget|my name is)\b"]),
    ("feedback", [r"\b(feedback|rate|review|improve)\b"]),
    ("smalltalk", [r"\b(how are you|what.?s up|bored|joke|fun fact)\b"]),
    ("question", [r"\?$|\b(what|why|how|when|where|who|which|can you|could you|explain|tell me)\b"]),
]

_ENTITY_PATTERNS: list[tuple[str, str]] = [
    ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("URL", r"https?://[^\s]+"),
    ("PHONE", r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b"),
    ("DATE", r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,?\s*\d{4})?)\b"),
    ("NUMBER", r"\b\d+(?:\.\d+)?%?\b"),
    ("PERSON", r"\b(?:my name is|i am|i'm|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"),
]


def _sentiment(text: str) -> str:
    if not text or not _analyzer:
        return "neutral"
    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.25:
        return "positive"
    if compound <= -0.25:
        return "negative"
    return "neutral"


def _intent(text: str) -> str:
    lowered = text.lower().strip()
    for name, patterns in _INTENT_PATTERNS:
        for pat in patterns:
            if re.search(pat, lowered, flags=re.IGNORECASE):
                return name
    return "general"


def _entities(text: str) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for label, pattern in _ENTITY_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = match.group(1) if match.lastindex else match.group(0)
            value = value.strip()
            key = f"{label}:{value.lower()}"
            if value and key not in seen:
                seen.add(key)
                found.append({"type": label, "value": value})
    return found[:20]


def analyze_text(text: str) -> dict[str, Any]:
    """Return sentiment, intent and entities for a user message."""
    text = (text or "").strip()
    return {
        "sentiment": _sentiment(text),
        "intent": _intent(text),
        "entities": _entities(text),
    }
