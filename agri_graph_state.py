"""
agri_graph_state.py
===================
LangGraph State Schema for the Farmer Field Operations Agent.

This TypedDict travels through every node of the LangGraph graph.
Each node reads from it and writes its outputs back into it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


# ============================================================
# LANGGRAPH STATE SCHEMA
# ============================================================

class AgriState(TypedDict, total=False):
    """
    Shared state that flows through every LangGraph node.

    Inputs (set by caller / Streamlit):
    ------------------------------------
    farmer_context   : Dict  – farmer + plot details collected from form
    latitude         : float – GPS latitude from map picker
    longitude        : float – GPS longitude from map picker
    farmer_question  : str   – The question the farmer is asking this turn
    chat_history     : List  – Previous (role, content) pairs for memory

    Node outputs (written by each node):
    ------------------------------------
    weather_raw      : Dict  – Raw Open-Meteo JSON response
    weather_summary  : Dict  – Processed weather statistics
    expanded_query   : str   – Query after agricultural term expansion
    detected_terms   : Dict  – Crops / states / topics detected
    retrieval_result : Dict  – Full retrieval bundle (chunks, citations …)
    ml_result        : Dict  – XGBoost model predictions
    groq_prompt      : str   – Fully assembled Groq prompt
    final_report     : str   – Groq LLM answer / report
    error            : str   – Any error message (stops the graph early)
    report_saved     : str   – Path where report was persisted
    node_logs        : List  – Human-readable log lines (shown in UI)
    """

    # ── Inputs ──────────────────────────────────────────────
    farmer_context:   Dict[str, Any]
    latitude:         Optional[float]
    longitude:        Optional[float]
    farmer_question:  str
    chat_history:     List[Dict[str, str]]   # [{"role": "...", "content": "..."}]

    # ── Node outputs ─────────────────────────────────────────
    weather_raw:      Dict[str, Any]
    weather_summary:  Dict[str, Any]
    expanded_query:   str
    detected_terms:   Dict[str, Any]
    retrieval_result: Dict[str, Any]
    ml_result:        Dict[str, Any]
    groq_prompt:      str
    final_report:     str
    error:            Optional[str]
    report_saved:     Optional[str]
    node_logs:        List[str]
