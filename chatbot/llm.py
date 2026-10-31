"""LLM reply generation via OpenAI-compatible endpoint (Groq by default)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

# Lazy client so missing key does not crash import-time
_client: OpenAI | None = None


def _get_client() -> OpenAI | None:
    global _client
    if _client is not None:
        return _client
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("GROQ_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.groq.com/openai/v1"
    _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


SYSTEM_BASE = """You are a helpful, concise, and friendly Dynamic AI Chatbot.
- Answer clearly and accurately.
- When the user provides document context, base your answer primarily on that context and cite pages/slides/sheets when possible using the format [PDF page: N] or [Slide N] or [Sheet: name].
- Respect the requested response language.
- Use saved user memories when relevant (e.g. name, preferences).
- If you do not know something, say so honestly.
- Keep answers focused; use markdown for structure when helpful.
- For any code, always use fenced markdown code blocks with a language tag, e.g. ```python ... ```. Never put each line of code in separate inline backticks. Keep code line order correct and complete."""


def _current_time_instruction() -> str:
    """Inject real clock so the model can answer 'what time is it?' correctly."""
    utc_now = datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        ist = utc_now.astimezone(ZoneInfo("Asia/Kolkata"))
        local_line = ist.strftime("%A, %d %B %Y, %I:%M:%S %p IST (Asia/Kolkata)")
    except Exception:
        # Fallback if zoneinfo data missing
        local_line = utc_now.strftime("%A, %d %B %Y, %H:%M:%S UTC") + " (convert to local if needed)"
    utc_line = utc_now.strftime("%Y-%m-%d %H:%M:%S UTC")
    return (
        f"Current real date and time (authoritative): {local_line}. "
        f"UTC reference: {utc_line}. "
        "CRITICAL: If the user asks the current time, date, day, or 'what time is it', "
        "answer ONLY with this clock. Prefer IST for users in India. "
        "Never invent a different time. Never label IST time as UTC."
    )




def _language_instruction(language: str) -> str:
    language = (language or "auto").lower()
    mapping = {
        "english": "Respond entirely in clear English.",
        "hindi": "Respond entirely in Hindi using Devanagari script (हिन्दी).",
        "hinglish": "Respond in Hinglish: natural mix of Hindi in Roman script and English.",
        "marathi": "Respond entirely in Marathi using Devanagari script (मराठी).",
        "gujarati": "Respond entirely in Gujarati script (ગુજરાતી).",
        "bengali": "Respond entirely in Bengali script (বাংলা).",
        "tamil": "Respond entirely in Tamil script (தமிழ்).",
        "telugu": "Respond entirely in Telugu script (తెలుగు).",
        "kannada": "Respond entirely in Kannada script (ಕನ್ನಡ).",
        "malayalam": "Respond entirely in Malayalam script (മലയാളം).",
        "punjabi": "Respond entirely in Punjabi using Gurmukhi script (ਪੰਜਾਬੀ).",
        "urdu": "Respond entirely in Urdu using Arabic script (اردو).",
        "spanish": "Respond entirely in Spanish (Español).",
        "french": "Respond entirely in French (Français).",
        "german": "Respond entirely in German (Deutsch).",
        "arabic": "Respond entirely in Arabic script (العربية).",
        "chinese": "Respond entirely in Simplified Chinese (中文).",
        "japanese": "Respond entirely in Japanese (日本語).",
    }
    if language in mapping:
        return mapping[language]
    return (
        "Match the language of the user's latest message. "
        "Support English, Hindi, Hinglish, other Indian languages, and major world languages."
    )


def _build_messages(
    user_message: str,
    history: list[dict],
    analysis: dict,
    language: str,
    memories: list[str],
    document_context: str,
) -> list[dict[str, str]]:
    system_parts = [SYSTEM_BASE, _current_time_instruction(), _language_instruction(language)]

    if analysis:
        system_parts.append(
            f"Detected user sentiment: {analysis.get('sentiment', 'neutral')}. "
            f"Detected intent: {analysis.get('intent', 'general')}."
        )
        if analysis.get("sentiment") == "negative":
            system_parts.append("Be extra empathetic and supportive.")

    if memories:
        mem_block = "\n".join(f"- {m}" for m in memories[:15])
        system_parts.append(
            "Long-term memories about this user (use only when relevant):\n" + mem_block
        )

    if document_context:
        doc_rules = (
            "DOCUMENT CONTEXT below may include one or more files. "
            "Rules: (1) By default focus on the most recently attached document. "
            "(2) If the user asks about a previous, earlier, last, or other file, use that document from the context. "
            "(3) If multiple documents are present, say which file the answer comes from. "
            "(4) Do not invent facts that contradict the document. "
            "(5) If the document is an assignment / worksheet that lists QUESTIONS or case studies "
            "and the user asks for answers, solutions, or help, YOU SHOULD ANSWER those questions "
            "using the case details in the document plus solid subject knowledge. "
            "Do NOT refuse with only 'the document has no answers'. "
            "Write clear, structured student-ready answers for each question. "
            "(6) Only say information is missing when the user asks to quote something that is truly not in the file."
        )
        system_parts.append(doc_rules + chr(10) + chr(10) + document_context[:100000])


    messages: list[dict[str, str]] = [
        {"role": "system", "content": "\n\n".join(system_parts)}
    ]

    # With an active document, keep less history so older PDF answers do not dominate
    history_window = history[-2:] if document_context else history[-10:]
    for item in history_window:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    if document_context:
        messages.append({
            "role": "system",
            "content": (
                "Reminder: Answer using ONLY the ACTIVE DOCUMENT CONTEXT in the system prompt. "
                "Do not reuse content from earlier documents discussed in this chat."
            ),
        })

    messages.append({"role": "user", "content": user_message})
    return messages




def _is_time_query(message: str) -> bool:
    t = (message or "").lower().strip()
    keys = [
        "what time", "what's the time", "whats the time", "current time",
        "time now", "time is it", "abhi time", "abhi kitna baja", "kitna baja",
        "what is the time", "tell me the time", "date today", "what day is",
        "today's date", "todays date", "aaj ki date", "aaj ka din",
    ]
    return any(k in t for k in keys)


def _direct_time_answer(language: str = "auto") -> str:
    utc_now = datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        ist = utc_now.astimezone(ZoneInfo("Asia/Kolkata"))
    except Exception:
        ist = utc_now
    # Format
    day = ist.strftime("%A")
    date = ist.strftime("%d %B %Y")
    clock = ist.strftime("%I:%M %p").lstrip("0")
    utc_clock = utc_now.strftime("%H:%M UTC")
    lang = (language or "auto").lower()
    if lang == "hindi":
        return f"अभी का समय है **{clock} IST** ({day}, {date})। (UTC: {utc_clock})"
    if lang == "hinglish":
        return f"Abhi time hai **{clock} IST** — {day}, {date}. (UTC: {utc_clock})"
    return (
        f"The current time is **{clock} IST** (India Standard Time).\n"
        f"Date: **{day}, {date}**.\n"
        f"UTC reference: {utc_clock}."
    )


def generate_reply(
    message: str,
    history: list[dict] | None = None,
    analysis: dict | None = None,
    language: str = "auto",
    memories: list[str] | None = None,
    document_context: str = "",
    document_filename: str | None = None,
) -> tuple[str, str]:
    """
    Generate an assistant reply.
    Returns (reply_text, source_label).
    Falls back to a simple rule-based response if the LLM is unavailable.
    """
    history = history or []
    analysis = analysis or {}
    memories = memories or []

    # Answer time/date questions with the real system clock (IST)
    if _is_time_query(message):
        return _direct_time_answer(language), "clock"

    client = _get_client()
    model = os.getenv("GROQ_MODEL") or os.getenv("OPENAI_MODEL") or "openai/gpt-oss-20b"

    if client is None:
        return _fallback_reply(message, analysis, language), "rule-based (no API key)"

    try:
        messages = _build_messages(
            message, history, analysis, language, memories, document_context or ""
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            max_tokens=1500,
        )
        reply = (response.choices[0].message.content or "").strip()
        if not reply:
            reply = "I could not generate a response. Please try again."
        source = "Groq LLM"
        if document_context and document_filename:
            source = document_filename
        return reply, source
    except Exception as exc:  # noqa: BLE001
        # Keep the user experience alive
        fallback = _fallback_reply(message, analysis, language)
        return (
            f"{fallback}\n\n_(Note: AI service temporarily unavailable: {type(exc).__name__})_",
            "fallback",
        )


def _fallback_reply(message: str, analysis: dict, language: str) -> str:
    intent = (analysis or {}).get("intent", "general")
    sentiment = (analysis or {}).get("sentiment", "neutral")

    if language == "hindi":
        if intent == "greeting":
            return "नमस्ते! मैं आपका AI सहायक हूँ। आज मैं आपकी कैसे मदद कर सकता हूँ?"
        if intent == "thanks":
            return "आपका स्वागत है! और कोई मदद चाहिए तो बताएँ।"
        return "मैं अभी सीमित मोड में हूँ। कृपया बाद में पुनः प्रयास करें या अपना प्रश्न सरल बनाएँ।"

    if language == "hinglish":
        if intent == "greeting":
            return "Hey! Main aapka AI assistant hoon. Aaj main aapki kaise madad kar sakta hoon?"
        return "Abhi main limited mode mein hoon. Thodi der baad try karein ya sawal simple banayein."

    if intent == "greeting":
        return "Hello! I'm your Dynamic AI Chatbot. How can I help you today?"
    if intent == "thanks":
        return "You're welcome! Let me know if there's anything else I can help with."
    if intent == "farewell":
        return "Goodbye! Feel free to come back anytime."
    if sentiment == "negative":
        return (
            "I'm sorry you're feeling that way. I'm here to help — "
            "could you tell me a bit more about what you need?"
        )
    return (
        "I received your message. The full AI model is currently unavailable, "
        "but you can still ask questions and I will do my best with built-in knowledge."
    )
