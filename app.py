import os
import uuid
import re
import json
from pathlib import Path

from werkzeug.utils import secure_filename
from pypdf import PdfReader
import pandas as pd
from docx import Document as DocxDocument
from pptx import Presentation
from io import BytesIO
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

from werkzeug.security import generate_password_hash, check_password_hash

from chatbot.database import (
    init_db, save_message, get_history, save_feedback, analytics_summary,
    create_conversation, list_conversations, conversation_exists, user_owns_conversation,
    rename_conversation, delete_conversation,
    save_memory, list_memories, get_memories, delete_memory,
    create_user, get_user_by_username, get_user_by_id,
)
from chatbot.nlp import analyze_text
from chatbot.llm import generate_reply

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-development-secret")
init_db()
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB per request

from werkzeug.exceptions import RequestEntityTooLarge

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({"error": "File is too large. Maximum size is 10 MB."}), 413

PDF_STORE = Path(app.instance_path) / "pdf_uploads"
PDF_STORE.mkdir(parents=True, exist_ok=True)
MAX_PDF_TEXT_CHARS = 120_000


def _new_conversation_id():
    return str(uuid.uuid4())


def _memory_user_id():
    """Logged-in user id, or anonymous browser session id."""
    if session.get("user_id"):
        return f"user:{session['user_id']}"
    if not session.get("memory_user_id"):
        session["memory_user_id"] = str(uuid.uuid4())
    return session["memory_user_id"]



def _conversation_owner_id() -> str | None:
    """Logged-in users get account-scoped chats; guests share the anonymous pool."""
    uid = session.get("user_id")
    if uid:
        return f"user:{uid}"
    return None

def _current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    return get_user_by_id(int(uid))


def _pdf_user_dir():
    # Windows cannot use ":" in folder names (e.g. user:3) — sanitize for the filesystem
    user_id = _memory_user_id()
    safe_id = (
        str(user_id)
        .replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )
    directory = PDF_STORE / safe_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _session_documents():
    documents = session.get("pdf_documents", [])
    return documents if isinstance(documents, list) else []


def _capture_explicit_memory(message):
    """Capture only clear memory requests or simple name statements; avoid broad auto-saving."""
    text = " ".join((message or "").strip().split())
    explicit_patterns = [
        r"^(?:please\s+)?remember that\s+(.+)$",
        r"^(?:please\s+)?remember\s+(.+)$",
        r"^yaad rakhna ki\s+(.+)$",
        r"^yaad rakhna\s+(.+)$",
    ]
    for pattern in explicit_patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            memory = match.group(1).strip().rstrip(".! ")
            if memory and len(memory) <= 500:
                return memory, "explicit"

    name_patterns = [
        r"^(?:my name is|i am called|call me)\s+([A-Za-z][A-Za-z .'-]{0,60})[.! ]*$",
        r"^(?:mera naam)\s+([A-Za-z][A-Za-z .'-]{0,60})\s*(?:hai)?[.! ]*$",
    ]
    for pattern in name_patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip().rstrip(".! ")
            if name:
                return f"The user's name is {name}.", "profile"
    return None, None


@app.get("/")
def index():
    conversation_id = session.get("conversation_id")
    if not conversation_id or not conversation_exists(conversation_id):
        conversation_id = _new_conversation_id()
        create_conversation(conversation_id, owner_id=_conversation_owner_id())
        session["conversation_id"] = conversation_id
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Please enter a message."}), 400
    if len(message) > 4000:
        return jsonify({"error": "Message is too long (maximum 4000 characters)."}), 400

    conversation_id = session.get("conversation_id")
    if not conversation_id or not conversation_exists(conversation_id):
        conversation_id = _new_conversation_id()
        create_conversation(conversation_id, owner_id=_conversation_owner_id())
        session["conversation_id"] = conversation_id

    language = (data.get("language") or "auto").strip().lower()
    allowed_languages = {
        "auto", "english", "hindi", "hinglish",
        "marathi", "gujarati", "bengali", "tamil", "telugu",
        "kannada", "malayalam", "punjabi", "urdu",
        "spanish", "french", "german", "arabic", "chinese", "japanese",
    }
    if language not in allowed_languages:
        language = "auto"

    document_ids = data.get("document_ids") or ([data.get("document_id")] if data.get("document_id") else [])
    if isinstance(document_ids, str):
        document_ids = [document_ids]
    document_ids = [str(item) for item in document_ids if item][:10]

    # If user asks about previous / earlier / other files, include all session documents
    msg_l = (message or "").lower()
    wants_previous = any(k in msg_l for k in (
        "previous file", "previous pdf", "previous document", "earlier file",
        "earlier pdf", "earlier document", "last file", "last pdf", "last document",
        "other file", "other pdf", "other document", "both file", "both pdf",
        "both document", "all file", "all pdf", "all document", "previous one",
        "earlier one", "from before", "old file", "old pdf", "old document",
        "pehle wala", "pichla", "previous upload",
    ))
    session_docs = _session_documents()
    session_ids = [str(d.get("id")) for d in session_docs if d.get("id")]
    active_id = str(session.get("active_document_id") or "")

    wants_new = any(k in msg_l for k in (
        "new file", "new pdf", "new document", "latest file", "latest pdf",
        "latest document", "current file", "current pdf", "current document",
        "just uploaded", "i uploaded", "the new one", "naya file", "naya pdf",
    ))


    # Match a specific uploaded file by name / course code (strict — avoid matching both assignments)
    def _score_doc_match(message_l: str, doc_name: str) -> int:
        """Higher score = stronger match. 0 = no match."""
        name_l = (doc_name or "").lower()
        stem = Path(doc_name or "").stem.lower()
        score = 0
        if name_l and name_l in message_l:
            score = max(score, 100)
        if stem and len(stem) >= 5 and stem in message_l:
            score = max(score, 90)
        stem_sp = stem.replace("_", " ").replace("-", " ")
        if stem_sp and len(stem_sp) >= 5 and stem_sp in message_l:
            score = max(score, 90)
        compact = re.sub(r"[^a-z0-9]", "", stem)
        for m in re.finditer(r"\bbs\s*[-_]?\s*(\d{3})\b", message_l):
            code = m.group(1)
            if code and code in name_l:
                score = max(score, 80)
            if code and code in compact:
                score = max(score, 80)
        for m in re.finditer(r"\b(\d{3})\b", message_l):
            code = m.group(1)
            if code and re.search(r"(?:^|[^0-9])" + re.escape(code) + r"(?:[^0-9]|$)", name_l):
                score = max(score, 70)
        return score

    scored = []
    for d in session_docs:
        sc = _score_doc_match(msg_l, str(d.get("name") or ""))
        if sc > 0:
            scored.append((sc, str(d.get("id")), str(d.get("name") or "")))
    scored.sort(key=lambda x: (-x[0],))

    if scored:
        # Use only the best match so BS501 and BS502 are not mixed
        document_ids = [scored[0][1]]
    elif wants_previous:
        for sid in session_ids:
            if sid not in document_ids:
                document_ids.append(sid)
        document_ids = document_ids[:10]
    elif wants_new and session_ids:
        document_ids = [session_ids[-1]]
    elif not document_ids:
        if active_id and active_id in session_ids:
            document_ids = [active_id]
        elif session_ids:
            document_ids = [session_ids[-1]]

    document_context_parts = []
    document_names = []
    available_documents = _session_documents()
    for document_id in document_ids:
        selected = next((d for d in available_documents if d.get("id") == document_id), None)
        if not selected:
            return jsonify({"error": "A selected document is no longer available. Please upload it again."}), 400
        text_path = _pdf_user_dir() / f"{document_id}.txt"
        if not text_path.exists():
            return jsonify({"error": f"Stored text for {selected.get('name', 'document')} was not found. Please upload it again."}), 400
        content = text_path.read_text(encoding="utf-8", errors="replace")
        document_context_parts.append(f"\n\n===== DOCUMENT: {selected.get('name', 'Uploaded document')} =====\n{content[:40000]}")
        document_names.append(selected.get("name", "Uploaded document"))
    document_context = "\n".join(document_context_parts)[:120000]
    document_filename = ", ".join(document_names) if document_names else None

    analysis = analyze_text(message)
    memory_user_id = _memory_user_id()
    captured_memory, memory_category = _capture_explicit_memory(message)
    memory_saved = False
    if captured_memory:
        memory_saved = save_memory(memory_user_id, captured_memory, memory_category)

    history = get_history(conversation_id, limit=12)
    memories = get_memories(memory_user_id, limit=20)

    # When a document is active, drop prior chat so another file's answers cannot leak in
    history_for_llm = [] if document_context else history

    if document_context and document_filename:
        document_context = (
            "PRIMARY FILE FOR THIS ANSWER: " + str(document_filename) + chr(10)
            + "Answer using ONLY this file. Ignore any other assignment discussed earlier."
            + chr(10) + chr(10)
            + document_context
        )

    reply, source = generate_reply(
        message,
        history_for_llm,
        analysis,
        language=language,
        memories=memories,
        document_context=document_context,
        document_filename=document_filename,
    )

    # Use source names and validate any PDF page citations against available context.
    if document_context and document_filename:
        cited_pages = sorted(set(re.findall(r"\[PDF page:\s*(\d+)\]", reply, flags=re.IGNORECASE)))
        if cited_pages:
            source = f"{document_filename} - PDF page(s) {', '.join(cited_pages)}"
        else:
            source = document_filename

    save_message(conversation_id, "user", message, analysis, source=None)
    assistant_message_id = save_message(
        conversation_id, "assistant", reply, analysis, source=source
    )

    return jsonify({
        "reply": reply,
        "sentiment": analysis["sentiment"],
        "intent": analysis["intent"],
        "entities": analysis["entities"],
        "source": source,
        "conversation_id": conversation_id,
        "assistant_message_id": assistant_message_id,
        "memory_saved": bool(captured_memory and memory_saved),
    })


@app.post("/api/new-chat")
def new_chat():
    conversation_id = _new_conversation_id()
    create_conversation(conversation_id, owner_id=_conversation_owner_id())
    session["conversation_id"] = conversation_id
    return jsonify({"ok": True, "conversation_id": conversation_id})


@app.get("/api/history")
def history():
    conversation_id = session.get("conversation_id")
    if conversation_id and not user_owns_conversation(conversation_id, _conversation_owner_id()):
        session.pop("conversation_id", None)
        conversation_id = None
    return jsonify({
        "conversation_id": conversation_id,
        "messages": get_history(conversation_id, limit=500) if conversation_id else []
    })


@app.get("/api/conversations")
def conversations():
    return jsonify({
        "conversations": list_conversations(limit=100, owner_id=_conversation_owner_id()),
        "current_conversation_id": session.get("conversation_id")
    })


@app.post("/api/conversations/<conversation_id>/open")
def open_conversation(conversation_id):
    if not conversation_exists(conversation_id):
        return jsonify({"error": "Conversation not found."}), 404
    if not user_owns_conversation(conversation_id, _conversation_owner_id()):
        return jsonify({"error": "Conversation not found."}), 404
    session["conversation_id"] = conversation_id
    return jsonify({
        "ok": True,
        "conversation_id": conversation_id,
        "messages": get_history(conversation_id, limit=500)
    })


@app.patch("/api/conversations/<conversation_id>")
def update_conversation(conversation_id):
    if not user_owns_conversation(conversation_id, _conversation_owner_id()):
        return jsonify({"error": "Conversation not found."}), 404
    data = request.get_json(silent=True) or {}
    title = data.get("title", "")
    try:
        updated = rename_conversation(conversation_id, title)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    if not updated:
        return jsonify({"error": "Conversation not found."}), 404
    return jsonify({"ok": True})


@app.delete("/api/conversations/<conversation_id>")
def remove_conversation(conversation_id):
    if not user_owns_conversation(conversation_id, _conversation_owner_id()):
        return jsonify({"error": "Conversation not found."}), 404
    deleted = delete_conversation(conversation_id)
    if not deleted:
        return jsonify({"error": "Conversation not found."}), 404

    if session.get("conversation_id") == conversation_id:
        new_id = _new_conversation_id()
        create_conversation(new_id, owner_id=_conversation_owner_id())
        session["conversation_id"] = new_id
        return jsonify({"ok": True, "new_conversation_id": new_id})
    return jsonify({"ok": True})


@app.post("/api/feedback")
def feedback():
    data = request.get_json(silent=True) or {}
    message_id = data.get("message_id")
    rating = data.get("rating")
    comment = (data.get("comment") or "")[:1000]
    if not isinstance(message_id, int) or rating not in ("up", "down"):
        return jsonify({
            "error": "message_id and rating ('up' or 'down') are required."
        }), 400
    try:
        save_feedback(message_id, rating, comment)
    except ValueError as error:
        return jsonify({"error": str(error)}), 404
    return jsonify({"ok": True})


@app.get("/api/analytics")
def analytics():
    return jsonify(analytics_summary())


@app.get("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.get("/api/memories")
def memories():
    return jsonify({"memories": list_memories(_memory_user_id(), limit=100)})


@app.delete("/api/memories/<int:memory_id>")
def remove_memory(memory_id):
    deleted = delete_memory(_memory_user_id(), memory_id)
    if not deleted:
        return jsonify({"error": "Memory not found."}), 404
    return jsonify({"ok": True})


ALLOWED_EXTENSIONS = {"pdf", "docx", "xlsx", "xls", "csv", "txt", "pptx", "jpg", "jpeg", "png", "webp", "gif"}
MAX_DOCUMENT_CHARS = 40000

def _extract_document(uploaded, extension):
    """Extract searchable text from supported office/document formats."""
    if extension == "pdf":
        reader = PdfReader(uploaded.stream)
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    raise ValueError("This PDF is password-protected. Upload an unlocked PDF.")
            except Exception as exc:
                raise ValueError("This PDF is password-protected. Upload an unlocked PDF.") from exc
        parts = []
        for number, page in enumerate(reader.pages, 1):
            text = (page.extract_text() or "").strip()
            if text:
                parts.append(f"--- PDF PAGE {number} ---\n{text}")
        return "\n\n".join(parts), len(reader.pages), None
    raw = uploaded.read()
    if extension == "txt":
        return raw.decode("utf-8-sig", errors="replace"), None, None
    if extension == "docx":
        doc = DocxDocument(BytesIO(raw))
        parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table_no, table in enumerate(doc.tables, 1):
            parts.append(f"[Table {table_no}]\n" + "\n".join(" | ".join(c.text.strip() for c in row.cells) for row in table.rows))
        return "\n\n".join(parts), None, None
    if extension == "pptx":
        prs = Presentation(BytesIO(raw)); parts=[]
        for i, slide in enumerate(prs.slides, 1):
            lines=[]
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip(): lines.append(shape.text.strip())
                if getattr(shape, "has_table", False):
                    lines.extend(" | ".join(cell.text.strip() for cell in row.cells) for row in shape.table.rows)
            if lines: parts.append(f"--- SLIDE {i} ---\n" + "\n".join(lines))
        return "\n\n".join(parts), len(prs.slides), None
    if extension in {"csv", "xlsx", "xls"}:
        if extension == "csv":
            df = pd.read_csv(BytesIO(raw), nrows=5000)
            sheets = {"CSV": df}
        else:
            sheets = pd.read_excel(BytesIO(raw), sheet_name=None, nrows=5000)
        parts=[]; analysis=[]
        for sheet_name, df in sheets.items():
            df = df.copy()
            parts.append(f"--- SHEET: {sheet_name} | rows={len(df)} | columns={len(df.columns)} ---\nColumns: {', '.join(map(str, df.columns))}\n" + df.head(100).to_csv(index=False))
            numeric = df.select_dtypes(include="number")
            summary = [f"Sheet: {sheet_name}", f"Rows loaded: {len(df)}", f"Columns: {len(df.columns)}", f"Missing cells: {int(df.isna().sum().sum())}", f"Duplicate rows: {int(df.duplicated().sum())}"]
            if len(numeric.columns):
                summary.append("Numeric summary (count/mean/median/min/max):\n" + numeric.describe().loc[["count","mean","min","max"]].to_string() + "\nMedians:\n" + numeric.median().to_string())
                corr = numeric.corr(numeric_only=True)
                if corr.shape[0] > 1: summary.append("Correlation matrix:\n" + corr.round(3).to_string())
            analysis.append("\n".join(summary))
        context = "\n\n".join(parts)
        computed = "\n\n===== COMPUTED SPREADSHEET ANALYSIS (calculated from uploaded data) =====\n" + "\n\n".join(analysis)
        return context + computed, None, {"sheets": list(sheets.keys()), "rows": sum(len(d) for d in sheets.values())}
    raise ValueError("Unsupported file format.")

@app.post("/api/upload-pdf")
@app.post("/api/upload-document")
def upload_pdf():
    try:
        uploaded = request.files.get("file")
        if not uploaded or not uploaded.filename:
            return jsonify({"error": "Choose a document first."}), 400

        original_name = uploaded.filename
        safe_name = secure_filename(original_name) or "uploaded_document"
        extension = (
            Path(original_name).suffix.lower().lstrip(".")
            or Path(safe_name).suffix.lower().lstrip(".")
        )
        if not extension:
            return jsonify({"error": "File has no extension. Use PDF, DOCX, TXT, etc."}), 400
        if extension not in ALLOWED_EXTENSIONS:
            return jsonify({
                "error": f"Unsupported type .{extension}. Use: PDF, DOCX, PPTX, XLSX, CSV, TXT, JPG, PNG."
            }), 400
        if not Path(safe_name).suffix:
            safe_name = f"{safe_name}.{extension}"

        document_id = str(uuid.uuid4())

        if extension in {"jpg", "jpeg", "png", "webp", "gif"}:
            extracted_text = (
                "[Image attached: " + safe_name + "]\n"
                "This is an image file. Text-inside-image (OCR) is not enabled in this project. "
                "You can still refer to this uploaded photo in chat."
            )
            page_count, table_info = 1, None
        else:
            extracted_text, page_count, table_info = _extract_document(uploaded, extension)
            extracted_text = (extracted_text or "").strip()[:MAX_DOCUMENT_CHARS]

        if not extracted_text:
            return jsonify({
                "error": "No extractable text was found. Scanned PDFs need OCR (not included)."
            }), 400

        store_dir = _pdf_user_dir()
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / f"{document_id}.txt").write_text(extracted_text, encoding="utf-8")

        latest = {
            "id": document_id,
            "name": safe_name,
            "type": extension.upper(),
            "pages": page_count or 0,
            "characters": len(extracted_text),
            "table_info": table_info,
        }
        # Keep recent uploads (for "previous file" questions); latest is preferred by default
        documents = [d for d in list(_session_documents()) if d.get("id") != document_id]
        documents.append(latest)
        session["pdf_documents"] = documents[-5:]
        session["active_document_id"] = document_id
        session.modified = True

        return jsonify({
            "ok": True,
            "document": latest,
            "documents": session["pdf_documents"],
            "active_document_id": document_id,
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("upload failed")
        return jsonify({
            "error": f"Upload failed: {type(exc).__name__}: {exc}"
        }), 500


@app.get("/api/pdfs")
def list_pdfs():
    docs = _session_documents()
    active = session.get("active_document_id")
    if not active and docs:
        active = docs[-1].get("id")
    return jsonify({
        "documents": docs,
        "active_document_id": active,
    })


@app.delete("/api/pdfs/<document_id>")
def delete_pdf(document_id):
    documents = _session_documents()
    selected = next((d for d in documents if d.get("id") == document_id), None)
    if not selected:
        return jsonify({"error": "PDF not found."}), 404

    text_path = _pdf_user_dir() / f"{document_id}.txt"
    try:
        text_path.unlink(missing_ok=True)
    except OSError:
        app.logger.warning("Could not remove stored PDF text for %s", document_id)

    session["pdf_documents"] = [
        d for d in documents if d.get("id") != document_id
    ]
    session.modified = True
    return jsonify({"ok": True, "documents": session["pdf_documents"]})



@app.post("/api/register")
def register():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""
    display_name = (data.get("display_name") or username).strip()
    if len(username) < 3 or len(username) > 40:
        return jsonify({"error": "Username must be 3–40 characters."}), 400
    if not username.replace("_", "").isalnum():
        return jsonify({"error": "Username may only contain letters, numbers, and underscore."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400
    user = create_user(username, generate_password_hash(password), display_name)
    if not user:
        return jsonify({"error": "Username already taken."}), 409
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session.pop("memory_user_id", None)  # switch to account-scoped memory
    session.pop("conversation_id", None)  # per-user chat history
    session.pop("pdf_documents", None)
    session.pop("active_document_id", None)
    return jsonify({"ok": True, "user": {"id": user["id"], "username": user["username"], "display_name": user["display_name"]}})


@app.post("/api/login")
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""
    user = get_user_by_username(username)
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid username or password."}), 401
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session.pop("memory_user_id", None)
    session.pop("conversation_id", None)  # per-user chat history
    session.pop("pdf_documents", None)
    session.pop("active_document_id", None)
    return jsonify({
        "ok": True,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "display_name": user.get("display_name") or user["username"],
        },
    })


@app.post("/api/logout")
def logout():
    session.pop("user_id", None)
    session.pop("username", None)
    session.pop("conversation_id", None)
    session.pop("pdf_documents", None)
    session.pop("active_document_id", None)
    return jsonify({"ok": True})


@app.get("/api/me")
def me():
    user = _current_user()
    if not user:
        return jsonify({"authenticated": False, "user": None})
    return jsonify({
        "authenticated": True,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "display_name": user.get("display_name") or user["username"],
        },
    })


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "Dynamic AI Chatbot"})


if __name__ == "__main__":
    app.run(
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000"))
    )