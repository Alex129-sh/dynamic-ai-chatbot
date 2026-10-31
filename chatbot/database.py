"""SQLite persistence for conversations, messages, feedback, memories and analytics."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(
    os.getenv("CHATBOT_DB_PATH")
    or Path(__file__).resolve().parent.parent / "instance" / "chatbot.db"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()



def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New chat',
                owner_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sentiment TEXT,
                intent TEXT,
                entities_json TEXT,
                source TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(conversation_id)
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER NOT NULL,
                rating TEXT NOT NULL,
                comment TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (message_id) REFERENCES messages(id)
            );

            CREATE TABLE IF NOT EXISTS user_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                memory TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'note',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                display_name TEXT,
                created_at TEXT NOT NULL
            );
            """
        )

        # Migrate older databases before creating the index.
        columns = [
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(conversations)"
            ).fetchall()
        ]

        if "owner_id" not in columns:
            conn.execute(
                "ALTER TABLE conversations ADD COLUMN owner_id TEXT"
            )

        # Create indexes after the migration.
        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, created_at);

            CREATE INDEX IF NOT EXISTS idx_memories_user
                ON user_memories(user_id, created_at);

            CREATE INDEX IF NOT EXISTS idx_users_username
                ON users(username);

            CREATE INDEX IF NOT EXISTS idx_conversations_owner
                ON conversations(owner_id, updated_at);
            """
        )



def create_conversation(
    conversation_id: str,
    title: str = "New chat",
    owner_id: str | None = None,
) -> None:
    now = _utc_now()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO conversations
                (conversation_id, title, owner_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, title or "New chat", owner_id, now, now),
        )
        # If row already existed without owner, attach owner when provided
        if owner_id:
            conn.execute(
                """
                UPDATE conversations
                SET owner_id = COALESCE(owner_id, ?)
                WHERE conversation_id = ?
                """,
                (owner_id, conversation_id),
            )


def conversation_exists(conversation_id: str | None) -> bool:
    if not conversation_id:
        return False
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return row is not None


def user_owns_conversation(conversation_id: str | None, owner_id: str | None) -> bool:
    """True if conversation exists and belongs to this owner (or both guest)."""
    if not conversation_id:
        return False
    with get_connection() as conn:
        row = conn.execute(
            "SELECT owner_id FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if not row:
            return False
        conv_owner = row["owner_id"] or None
        user_owner = owner_id or None
        return conv_owner == user_owner


def list_conversations(
    limit: int = 100,
    owner_id: str | None = None,
) -> list[dict[str, Any]]:
    with get_connection() as conn:
        if owner_id:
            rows = conn.execute(
                """
                SELECT conversation_id, title, created_at, updated_at, owner_id
                FROM conversations
                WHERE owner_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (owner_id, limit),
            ).fetchall()
        else:
            # Anonymous: only conversations with no owner (legacy / guest)
            rows = conn.execute(
                """
                SELECT conversation_id, title, created_at, updated_at, owner_id
                FROM conversations
                WHERE owner_id IS NULL OR owner_id = ''
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def rename_conversation(conversation_id: str, title: str) -> bool:
    title = (title or "").strip()
    if not title:
        raise ValueError("Title cannot be empty.")
    if len(title) > 120:
        title = title[:120]
    with get_connection() as conn:
        cur = conn.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE conversation_id = ?
            """,
            (title, _utc_now(), conversation_id),
        )
        return cur.rowcount > 0


def delete_conversation(conversation_id: str) -> bool:
    with get_connection() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cur = conn.execute(
            "DELETE FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        )
        return cur.rowcount > 0


def save_message(
    conversation_id: str,
    role: str,
    content: str,
    analysis: dict | None = None,
    source: str | None = None,
) -> int:
    analysis = analysis or {}
    entities = analysis.get("entities") or []
    now = _utc_now()
    with get_connection() as conn:
        # Bump conversation updated_at and auto-title from first user message
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
            (now, conversation_id),
        )
        if role == "user":
            row = conn.execute(
                "SELECT title FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if row and (row["title"] or "").strip() in ("", "New chat"):
                auto_title = " ".join(content.strip().split())[:60] or "New chat"
                conn.execute(
                    "UPDATE conversations SET title = ? WHERE conversation_id = ?",
                    (auto_title, conversation_id),
                )

        cur = conn.execute(
            """
            INSERT INTO messages
                (conversation_id, role, content, sentiment, intent, entities_json, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                role,
                content,
                analysis.get("sentiment"),
                analysis.get("intent"),
                json.dumps(entities, ensure_ascii=False),
                source,
                now,
            ),
        )
        return int(cur.lastrowid)


def get_history(conversation_id: str | None, limit: int = 50) -> list[dict[str, Any]]:
    if not conversation_id:
        return []
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, role, content, sentiment, intent, entities_json, source, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["entities"] = json.loads(d.pop("entities_json") or "[]")
        except json.JSONDecodeError:
            d["entities"] = []
        result.append(d)
    return result


def save_feedback(message_id: int, rating: str, comment: str = "") -> None:
    if rating not in ("up", "down"):
        raise ValueError("rating must be 'up' or 'down'")
    with get_connection() as conn:
        exists = conn.execute(
            "SELECT 1 FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if not exists:
            raise ValueError("Message not found.")
        conn.execute(
            """
            INSERT INTO feedback (message_id, rating, comment, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (message_id, rating, comment or "", _utc_now()),
        )


def save_memory(user_id: str, memory: str, category: str = "note") -> bool:
    memory = (memory or "").strip()
    if not memory or not user_id:
        return False
    with get_connection() as conn:
        # Avoid exact duplicates for the same user
        existing = conn.execute(
            """
            SELECT 1 FROM user_memories
            WHERE user_id = ? AND lower(memory) = lower(?)
            LIMIT 1
            """,
            (user_id, memory),
        ).fetchone()
        if existing:
            return False
        conn.execute(
            """
            INSERT INTO user_memories (user_id, memory, category, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, memory[:500], category or "note", _utc_now()),
        )
        return True


def list_memories(user_id: str, limit: int = 100) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, memory, category, created_at
            FROM user_memories
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_memories(user_id: str, limit: int = 20) -> list[str]:
    """Return plain memory strings for injection into the LLM prompt."""
    items = list_memories(user_id, limit=limit)
    return [item["memory"] for item in items]


def delete_memory(user_id: str, memory_id: int) -> bool:
    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM user_memories WHERE id = ? AND user_id = ?",
            (memory_id, user_id),
        )
        return cur.rowcount > 0


def analytics_summary() -> dict[str, Any]:
    with get_connection() as conn:
        total_messages = conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"]
        total_conversations = conn.execute(
            "SELECT COUNT(*) AS c FROM conversations"
        ).fetchone()["c"]
        user_messages = conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE role = 'user'"
        ).fetchone()["c"]
        assistant_messages = conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE role = 'assistant'"
        ).fetchone()["c"]

        sentiment_rows = conn.execute(
            """
            SELECT sentiment, COUNT(*) AS c
            FROM messages
            WHERE role = 'user' AND sentiment IS NOT NULL
            GROUP BY sentiment
            """
        ).fetchall()
        sentiments = {r["sentiment"]: r["c"] for r in sentiment_rows}

        intent_rows = conn.execute(
            """
            SELECT intent, COUNT(*) AS c
            FROM messages
            WHERE role = 'user' AND intent IS NOT NULL
            GROUP BY intent
            ORDER BY c DESC
            LIMIT 12
            """
        ).fetchall()
        intents = [{"intent": r["intent"], "count": r["c"]} for r in intent_rows]

        feedback_rows = conn.execute(
            """
            SELECT rating, COUNT(*) AS c
            FROM feedback
            GROUP BY rating
            """
        ).fetchall()
        feedback = {r["rating"]: r["c"] for r in feedback_rows}

        recent = conn.execute(
            """
            SELECT date(created_at) AS day, COUNT(*) AS c
            FROM messages
            GROUP BY date(created_at)
            ORDER BY day DESC
            LIMIT 14
            """
        ).fetchall()
        daily = [{"day": r["day"], "count": r["c"]} for r in reversed(recent)]

        avg_len = conn.execute(
            """
            SELECT AVG(LENGTH(content)) AS avg_len
            FROM messages
            WHERE role = 'user'
            """
        ).fetchone()["avg_len"]

        registered_users = conn.execute(
            "SELECT COUNT(*) AS c FROM users"
        ).fetchone()["c"]

        memory_count = conn.execute(
            "SELECT COUNT(*) AS c FROM user_memories"
        ).fetchone()["c"]

        source_rows = conn.execute(
            """
            SELECT COALESCE(source, 'unknown') AS source, COUNT(*) AS c
            FROM messages
            WHERE role = 'assistant'
            GROUP BY COALESCE(source, 'unknown')
            ORDER BY c DESC
            LIMIT 8
            """
        ).fetchall()
        sources = [{"source": r["source"], "count": r["c"]} for r in source_rows]

        recent_conversations = conn.execute(
            """
            SELECT conversation_id, title, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            LIMIT 8
            """
        ).fetchall()
        recent_chats = [dict(r) for r in recent_conversations]

        peak = max(daily, key=lambda d: d["count"]) if daily else None

        up = int(feedback.get("up") or 0)
        down = int(feedback.get("down") or 0)
        fb_total = up + down
        satisfaction = round((up / fb_total) * 100, 1) if fb_total else None

        msgs_per_conv = (
            round(total_messages / total_conversations, 1)
            if total_conversations else 0
        )

    return {
        "total_messages": total_messages,
        "total_conversations": total_conversations,
        "user_messages": user_messages,
        "assistant_messages": assistant_messages,
        "sentiments": sentiments,
        "top_intents": intents,
        "feedback": feedback,
        "daily_activity": daily,
        "avg_user_message_length": round(float(avg_len or 0), 1),
        "registered_users": registered_users,
        "memory_count": memory_count,
        "sources": sources,
        "recent_conversations": recent_chats,
        "peak_day": peak,
        "satisfaction_rate": satisfaction,
        "messages_per_conversation": msgs_per_conv,
    }


# ---------------------------------------------------------------------------
# User accounts
# ---------------------------------------------------------------------------

def create_user(username: str, password_hash: str, display_name: str | None = None) -> dict[str, Any] | None:
    username = (username or "").strip().lower()
    if not username or not password_hash:
        return None
    display_name = (display_name or username).strip()[:80]
    with get_connection() as conn:
        try:
            cur = conn.execute(
                """
                INSERT INTO users (username, password_hash, display_name, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (username, password_hash, display_name, _utc_now()),
            )
            return {
                "id": int(cur.lastrowid),
                "username": username,
                "display_name": display_name,
            }
        except sqlite3.IntegrityError:
            return None


def get_user_by_username(username: str) -> dict[str, Any] | None:
    username = (username or "").strip().lower()
    if not username:
        return None
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, display_name, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, username, display_name, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None