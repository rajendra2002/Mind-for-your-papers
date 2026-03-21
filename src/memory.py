"""
Memory Manager — Short-term + Long-term memory per chat session.

Short-term: Last N turns in-memory (list of dicts).
Long-term:  Key facts extracted by LLM, persisted as JSON per session.
"""

import os
import json
import time
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

SESSIONS_DIR = Path("./chat_sessions")


class MemoryManager:
    def __init__(self, session_id: str, short_term_limit: int = 10):
        self.session_id = session_id
        self.short_term_limit = short_term_limit

        # Paths
        self.session_dir = SESSIONS_DIR / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.session_dir / "meta.json"
        # Global long-term memory (shared across all sessions)
        self.long_term_path = SESSIONS_DIR / "global_facts.json"

        # History persistence
        self.history_path = self.session_dir / "history.json"
        self.short_term: list[dict] = self._load_history()

        # Long-term (persisted)
        self.long_term_facts: list[str] = self._load_long_term()

        # Groq client for fact extraction
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"

        # Counter for extraction trigger
        self._turn_counter = 0

    # ------------------------------------------------------------------ #
    #  Session metadata
    # ------------------------------------------------------------------ #
    def save_meta(self, name: str):
        data = {
            "name": name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": self.session_id,
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_meta(self) -> dict:
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    # ------------------------------------------------------------------ #
    #  Short-term memory
    # ------------------------------------------------------------------ #
    def _load_history(self) -> list[dict]:
        if self.history_path.exists():
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_history(self):
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(self.short_term, f, indent=2)

    def get_messages(self) -> list[dict]:
        return self.short_term

    def add_turn(self, user_msg: str, assistant_msg: str):
        self.short_term.append({
            "role": "user",
            "content": user_msg,
        })
        self.short_term.append({
            "role": "assistant",
            "content": assistant_msg,
        })
        # Trim to limit (each turn = 2 entries)
        max_entries = self.short_term_limit * 2
        if len(self.short_term) > max_entries:
            self.short_term = self.short_term[-max_entries:]
        
        self._save_history()

        self._turn_counter += 1
        # Extract facts every 3 turns
        if self._turn_counter % 3 == 0:
            self._extract_and_store_facts()

    def get_short_term_context(self) -> str:
        if not self.short_term:
            return ""
        lines = []
        for msg in self.short_term:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Long-term memory
    # ------------------------------------------------------------------ #
    def _load_long_term(self) -> list[str]:
        if self.long_term_path.exists():
            with open(self.long_term_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_long_term(self):
        with open(self.long_term_path, "w", encoding="utf-8") as f:
            json.dump(self.long_term_facts, f, indent=2)

    def get_long_term_context(self) -> str:
        # Reload to ensure we have latest global facts
        self.long_term_facts = self._load_long_term()
        if not self.long_term_facts:
            return ""
        return "\n".join(f"- {fact}" for fact in self.long_term_facts)

    def _extract_and_store_facts(self):
        """Use LLM to extract key facts from recent conversation."""
        # Reload to ensure we have latest global facts
        self.long_term_facts = self._load_long_term()

        recent = self.get_short_term_context()
        if not recent:
            return

        prompt = f"""Extract key facts from this conversation that should be remembered long-term.
Focus on: user preferences, names, important decisions, specific data points.
Return ONLY a JSON array of short fact strings. If no important facts, return [].

Conversation:
{recent}

Existing facts (do not duplicate):
{json.dumps(self.long_term_facts)}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": "You extract key facts as a JSON array of strings. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content.strip()

            # Parse JSON from response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            new_facts = json.loads(content)
            if isinstance(new_facts, list):
                # Reload again to minimize race condition window
                current_facts = self._load_long_term()
                updated = False
                for fact in new_facts:
                    if fact and fact not in current_facts:
                        current_facts.append(str(fact))
                        updated = True
                
                if updated:
                    self.long_term_facts = current_facts
                    self._save_long_term()
        except Exception:
            pass  # Silently fail — don't break the chat

    # ------------------------------------------------------------------ #
    #  Combined context for injection
    # ------------------------------------------------------------------ #
    def get_full_context(self) -> str:
        parts = []
        lt = self.get_long_term_context()
        if lt:
            parts.append(f"=== Long-Term Memory ===\n{lt}")
        st = self.get_short_term_context()
        if st:
            parts.append(f"=== Recent Conversation ===\n{st}")
        return "\n\n".join(parts)


# ------------------------------------------------------------------ #
#  Session helpers (used by the UI)
# ------------------------------------------------------------------ #
def list_all_sessions() -> list[dict]:
    """Return list of session metadata dicts, sorted newest first."""
    sessions = []
    if not SESSIONS_DIR.exists():
        return sessions
    for d in SESSIONS_DIR.iterdir():
        if d.is_dir():
            meta_path = d / "meta.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    sessions.append(json.load(f))
    sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return sessions


def delete_session_data(session_id: str):
    """Delete a session's data directory."""
    import shutil
    session_dir = SESSIONS_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
