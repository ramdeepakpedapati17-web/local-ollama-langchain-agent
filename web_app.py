import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from agent import build_agent, chat_turn


app = Flask(__name__)
chain, save_history_to_disk = build_agent()
sessions_dir = Path("sessions")
sessions_dir.mkdir(exist_ok=True)
session_index_path = sessions_dir / "_index.json"


def _load_session_index():
    if not session_index_path.exists():
        return {}
    try:
        with session_index_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_session_index(index):
    with session_index_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=True, indent=2)


def _upsert_session(session_id: str, tag: str | None):
    now = datetime.now(timezone.utc).isoformat()
    index = _load_session_index()
    existing = index.get(session_id, {})
    label = (tag or "").strip() or existing.get("tag") or f"Session {session_id[:8]}"
    index[session_id] = {
        "tag": label,
        "updated_at": now,
    }
    _save_session_index(index)
    return label


def _list_sessions():
    index = _load_session_index()
    sessions = []
    for session_id, details in index.items():
        sessions.append(
            {
                "session_id": session_id,
                "tag": details.get("tag", f"Session {session_id[:8]}"),
                "updated_at": details.get("updated_at", ""),
            }
        )
    sessions.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return sessions


def _session_history(session_id: str):
    file_path = sessions_dir / f"{session_id}.json"
    if not file_path.exists():
        return []

    try:
        with file_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    cleaned = []
    for record in records:
        role = record.get("type")
        content = record.get("content", "")
        if role in {"human", "ai"} and isinstance(content, str):
            cleaned.append({"type": role, "content": content})
    return cleaned


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/sessions")
def api_sessions():
    return jsonify({"sessions": _list_sessions()})


@app.get("/api/sessions/<session_id>/history")
def api_session_history(session_id):
    return jsonify({"session_id": session_id, "history": _session_history(session_id)})


@app.post("/api/chat")
def api_chat():
    payload = request.get_json(silent=True) or {}
    user_input = (payload.get("message") or "").strip()
    session_id = (payload.get("session_id") or "").strip() or str(uuid.uuid4())
    tag = (payload.get("tag") or "").strip()

    if not user_input:
        return jsonify({"error": "message is required"}), 400

    try:
        saved_tag = _upsert_session(session_id, tag)
        answer = chat_turn(chain, save_history_to_disk, session_id, user_input)
        return jsonify({"session_id": session_id, "tag": saved_tag, "answer": answer})
    except Exception as exc:
        return jsonify({"error": f"agent request failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
