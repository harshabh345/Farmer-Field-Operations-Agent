"""
agri_memory.py
==============
Context Memory Manager for the Farmer Field Operations Agent.

Provides:
  - ConversationMemory   : per-session rolling window memory
  - MemoryStore          : manages multiple sessions (one per farmer/tab)

Sprint 2: Wires conversation history so every follow-up question
remembers what was discussed in earlier turns.
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# CONVERSATION MEMORY
# ============================================================

class ConversationMemory:
    """
    Rolling-window conversation memory for a single farmer session.

    Stores a fixed number of (role, content) turns.
    The entire history is injected into the Groq prompt by the
    reasoning_node so the LLM can answer follow-up questions
    consistently.

    Args:
        window_size : Max number of (user + assistant) turns to keep.
                      Older turns are dropped automatically.
        session_id  : Optional identifier for persistence.
    """

    def __init__(self, window_size: int = 10, session_id: str = "default"):
        self.window_size = window_size
        self.session_id  = session_id
        self._history: deque[Dict[str, str]] = deque(maxlen=window_size * 2)
        self.created_at  = datetime.now().isoformat()

    # ── Mutators ─────────────────────────────────────────────

    def add_user_turn(self, content: str) -> None:
        """Add a user message to history."""
        self._history.append({"role": "user", "content": content})

    def add_assistant_turn(self, content: str) -> None:
        """Add an assistant (Groq) response to history."""
        self._history.append({"role": "assistant", "content": content})

    def add_turn(self, user_message: str, assistant_response: str) -> None:
        """Convenience: add a complete (user → assistant) exchange."""
        self.add_user_turn(user_message)
        self.add_assistant_turn(assistant_response)

    def clear(self) -> None:
        """Wipe all history (start a fresh conversation)."""
        self._history.clear()

    # ── Accessors ────────────────────────────────────────────

    @property
    def history(self) -> List[Dict[str, str]]:
        """Return history as a list of {role, content} dicts."""
        return list(self._history)

    @property
    def turn_count(self) -> int:
        """Number of individual messages (not pairs) stored."""
        return len(self._history)

    def last_n_turns(self, n: int = 5) -> List[Dict[str, str]]:
        """Return the last n message pairs (2*n individual messages)."""
        all_msgs = list(self._history)
        return all_msgs[-(n * 2):]

    def format_for_prompt(self, max_turns: int = 10) -> str:
        """
        Format conversation history as a human-readable block
        for injection into the Groq prompt.
        """
        msgs = self.last_n_turns(max_turns)
        if not msgs:
            return ""
        lines = []
        for msg in msgs:
            role    = msg["role"].upper()
            content = msg["content"][:1000]  # truncate very long turns
            lines.append(f"[{role}]: {content}")
        return "\n\n".join(lines)

    # ── Serialization (for Streamlit session_state persistence) ──

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "window_size": self.window_size,
            "created_at": self.created_at,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMemory":
        mem = cls(
            window_size=data.get("window_size", 10),
            session_id=data.get("session_id", "default"),
        )
        mem.created_at = data.get("created_at", datetime.now().isoformat())
        for msg in data.get("history", []):
            mem._history.append(msg)
        return mem


# ============================================================
# MEMORY STORE (multi-session manager)
# ============================================================

class MemoryStore:
    """
    Manages one ConversationMemory per session.

    Useful when the Streamlit app needs to serve multiple
    simultaneous farmer sessions (each browser tab gets its
    own persistent session_state, but this class makes the
    logic explicit and testable).
    """

    def __init__(self, default_window_size: int = 10):
        self.default_window_size = default_window_size
        self._sessions: Dict[str, ConversationMemory] = {}

    def get_or_create(self, session_id: str) -> ConversationMemory:
        """Return the memory for session_id, creating it if absent."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(
                window_size=self.default_window_size,
                session_id=session_id,
            )
        return self._sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        """Wipe history for a session."""
        if session_id in self._sessions:
            self._sessions[session_id].clear()

    def delete_session(self, session_id: str) -> None:
        """Remove a session entirely."""
        self._sessions.pop(session_id, None)

    @property
    def active_sessions(self) -> List[str]:
        return list(self._sessions.keys())


# ============================================================
# STREAMLIT HELPERS
# ============================================================

def get_streamlit_memory(session_state) -> ConversationMemory:
    """
    Get (or create) a ConversationMemory stored inside
    Streamlit's st.session_state.

    Usage in streamlit_app.py:
        from agri_memory import get_streamlit_memory
        memory = get_streamlit_memory(st.session_state)
        memory.add_turn(user_msg, agent_response)
    """
    if "agri_memory" not in session_state:
        session_state["agri_memory"] = ConversationMemory(window_size=10)
    return session_state["agri_memory"]


def memory_to_chat_history(memory: ConversationMemory) -> List[Dict[str, str]]:
    """
    Convert ConversationMemory to the list format expected by
    AgriState["chat_history"] for the LangGraph reasoning_node.
    """
    return memory.history


# ============================================================
# SELF-TEST (Sprint 2 verification)
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SPRINT 2 — CONTEXT MEMORY TEST")
    print("=" * 70)

    mem = ConversationMemory(window_size=5, session_id="test_session")

    # Simulate 3 conversation turns
    turns = [
        ("What crop should I grow?",
         "Based on your black soil and Karnataka location, sugarcane is ideal."),
        ("What fertilizer should I use?",
         "For sugarcane in Karnataka black soil, apply 250 kg/ha urea split in 3 doses."),
        ("What about irrigation?",
         "Drip irrigation is recommended. Water every 3-4 days during dry spells."),
    ]

    for user, assistant in turns:
        mem.add_turn(user, assistant)

    print(f"\nTurn count      : {mem.turn_count}")
    print(f"History entries : {len(mem.history)}")
    print(f"\n--- Formatted for prompt ---\n")
    print(mem.format_for_prompt(max_turns=5))

    # Test serialization
    d = mem.to_dict()
    mem2 = ConversationMemory.from_dict(d)
    assert mem2.turn_count == mem.turn_count, "Serialization round-trip failed!"
    print("\n✅ Memory serialization round-trip passed.")

    # Test window eviction
    for i in range(10):
        mem.add_turn(f"Question {i}", f"Answer {i}")
    assert mem.turn_count <= 10, "Window eviction failed!"
    print(f"✅ Window eviction: {mem.turn_count} messages (max 10 expected).")

    print("\nAll memory tests passed.")
