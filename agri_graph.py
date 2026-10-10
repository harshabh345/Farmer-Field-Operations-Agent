"""
agri_graph.py
=============
LangGraph graph for the Farmer Field Operations Agent.

Architecture:
    START
      ↓
    weather_node          – fetch live Open-Meteo weather
      ↓
    retrieval_node        – hybrid RAG (BM25 + Dense + Metadata + RRF)
      ↓
    ml_node               – XGBoost yield + crop recommendation
      ↓
    reasoning_node        – assemble Groq prompt with memory
      ↓
    report_node           – call Groq, save report
      ↓
    END

Each node:
  - receives the full AgriState
  - does its work
  - returns a dict of keys it wants to update in the state

Sprint 1 focus: graph wiring + node implementation.
Memory injection is wired here (Sprint 2 fills agri_memory.py).
"""

from __future__ import annotations

import os
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from agri_graph_state import AgriState

# ── Project root (relative to this file) ────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
load_dotenv(PROJECT_ROOT / ".env")

ML_ROOT = PROJECT_ROOT / "ml_training"
MODEL_DIR = ML_ROOT / "models"
DATA_ROOT = PROJECT_ROOT / "Data"
EMBEDDING_DIR = DATA_ROOT / "embeddings"
VECTOR_STORE_DIR = DATA_ROOT / "vector_store"

EMBEDDING_FILE = EMBEDDING_DIR / "agri_embeddings.npy"
RECORDS_FILE = EMBEDDING_DIR / "agri_embedding_records.jsonl"
PRODUCTION_MODEL_FILE = MODEL_DIR / "agriculture_production_xgb.pkl"
CROP_RECOMMENDATION_MODEL_FILE = MODEL_DIR / "crop_recommendation_xgb.pkl"
CHROMA_COLLECTION_NAME = "agri_documents"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


# ============================================================
# LAZY-LOADED SINGLETONS
# (Retriever and ML models are heavy – load once, reuse)
# ============================================================

_retriever_singleton: Optional[Any] = None
_ml_models_singleton: Optional[Any] = None


def get_retriever():
    """Return (and lazily create) the AgriculturalRetriever singleton."""
    global _retriever_singleton
    if _retriever_singleton is None:
        # Import from the original module to reuse all the battle-tested code
        from agri_core import AgriculturalRetriever
        _retriever_singleton = AgriculturalRetriever(
            use_reranker=False,          # fast mode for web app
            dense_k=100,
            bm25_k=100,
            metadata_k=100,
            final_context_chunks=35,
            max_context_chars=70_000,
        )
    return _retriever_singleton


def get_ml_models():
    """Return (and lazily create) the AgriculturalModels singleton."""
    global _ml_models_singleton
    if _ml_models_singleton is None:
        from agri_core import AgriculturalModels
        _ml_models_singleton = AgriculturalModels()
    return _ml_models_singleton


# ============================================================
# NODE 1 – WEATHER NODE
# ============================================================

def weather_node(state: AgriState) -> Dict[str, Any]:
    """
    Fetch live weather from Open-Meteo for the farmer's GPS location.

    Reads  : state["latitude"], state["longitude"]
    Writes : weather_raw, weather_summary, node_logs
    """
    logs = list(state.get("node_logs", []))
    logs.append("[WeatherNode] Fetching live Open-Meteo weather...")

    lat = state.get("latitude")
    lon = state.get("longitude")

    if lat is None or lon is None:
        logs.append("[WeatherNode] No GPS coordinates — skipping weather fetch.")
        return {
            "weather_raw": {"_status": "skipped"},
            "weather_summary": {},
            "node_logs": logs,
        }

    try:
        from agri_core import fetch_open_meteo, weather_statistics
    except ImportError as ie:
        logs.append(f"[WeatherNode] agri_core import failed: {ie}")
        logs.append("[WeatherNode] Run: pip install -r requirements_new.txt")
        return {
            "weather_raw": {"_status": "failed", "_error": str(ie)},
            "weather_summary": {},
            "node_logs": logs,
            "error": f"Missing dependency: {ie}. Run: pip install -r requirements_new.txt",
        }

    try:
        weather_raw = fetch_open_meteo(lat, lon, forecast_days=16)
        weather_summary = weather_statistics(weather_raw)
        status = weather_raw.get("_status", "unknown")
        logs.append(f"[WeatherNode] Weather fetched. Status: {status}")
        return {
            "weather_raw": weather_raw,
            "weather_summary": weather_summary,
            "node_logs": logs,
        }
    except Exception as exc:
        logs.append(f"[WeatherNode] Weather fetch error: {exc}")
        return {
            "weather_raw": {"_status": "failed", "_error": str(exc)},
            "weather_summary": {},
            "node_logs": logs,
            "error": f"WeatherNode failed: {exc}",
        }


# ============================================================
# NODE 2 – RETRIEVAL NODE
# ============================================================

def retrieval_node(state: AgriState) -> Dict[str, Any]:
    """
    Run hybrid RAG retrieval (BM25 + Dense vector + Metadata + RRF).

    Weather was already fetched by weather_node — we pass it directly
    into the retrieval result instead of calling Open-Meteo again.

    Reads  : state["farmer_question"], state["farmer_context"],
             state["weather_raw"], state["weather_summary"]
    Writes : retrieval_result, expanded_query, detected_terms, node_logs
    """
    logs = list(state.get("node_logs", []))
    logs.append("[RetrievalNode] Starting hybrid RAG retrieval...")

    question       = state.get("farmer_question", "")
    farmer_context = state.get("farmer_context", {})
    weather_raw    = state.get("weather_raw", {})
    weather_summary = state.get("weather_summary", {})

    if not question:
        question = (
            "What should I plan now considering soil, weather, irrigation, "
            "crop suitability, disease risk, and expected yield?"
        )

    try:
        retriever = get_retriever()
        # Do NOT pass lat/lon — weather already fetched in weather_node.
        # We retrieve only text evidence here.
        result = retriever.retrieve(
            query=question,
            farmer_context=farmer_context,
            latitude=None,   # skip internal weather fetch
            longitude=None,
        )
        # Inject the already-fetched weather so downstream nodes see it
        result["weather_raw"]     = weather_raw
        result["weather_summary"] = weather_summary

        n_chunks = len(result.get("selected_results", []))
        logs.append(f"[RetrievalNode] Retrieved {n_chunks} evidence chunks.")
        return {
            "retrieval_result": result,
            "expanded_query":   result.get("expanded_query", question),
            "detected_terms":   result.get("detected_terms", {}),
            "node_logs":        logs,
        }
    except Exception as exc:
        logs.append(f"[RetrievalNode] Error: {exc}")
        return {
            "retrieval_result": {},
            "expanded_query":   question,
            "detected_terms":   {},
            "node_logs":        logs,
            "error":            f"RetrievalNode failed: {exc}",
        }


# ============================================================
# NODE 3 – ML NODE
# ============================================================

def ml_node(state: AgriState) -> Dict[str, Any]:
    """
    Run XGBoost models: yield prediction + crop recommendation.

    Reads  : state["farmer_context"], state["weather_summary"]
    Writes : ml_result, node_logs
    """
    logs = list(state.get("node_logs", []))
    logs.append("[MLNode] Running XGBoost models...")

    farmer_context = dict(state.get("farmer_context", {}))
    weather_summary = state.get("weather_summary", {})

    # Enrich farmer_context with live weather values if not already provided
    current_weather = weather_summary.get("current", {})
    farmer_context.setdefault("weather", {})
    farmer_context["weather"].update({
        "temperature": current_weather.get(
            "temperature_2m",
            farmer_context["weather"].get("temperature", 0.0)
        ),
        "humidity": current_weather.get(
            "relative_humidity_2m",
            farmer_context["weather"].get("humidity", 0.0)
        ),
        "precipitation": current_weather.get(
            "precipitation",
            farmer_context["weather"].get("precipitation", 0.0)
        ),
    })

    try:
        models = get_ml_models()
        ml_result = models.run_all(farmer_context)
        logs.append("✅ [MLNode] ML predictions complete.")
        return {
            "ml_result": ml_result,
            "node_logs": logs,
        }
    except Exception as exc:
        logs.append(f"⚠️  [MLNode] ML models unavailable: {exc}")
        return {
            "ml_result": {
                "production_model": {"status": "unavailable", "reason": str(exc)},
                "crop_recommendation_model": {"status": "unavailable", "reason": str(exc)},
            },
            "node_logs": logs,
        }


# ============================================================
# NODE 4 – REASONING NODE (prompt assembly + memory)
# ============================================================

def reasoning_node(state: AgriState) -> Dict[str, Any]:
    """
    Assemble the full Groq prompt, injecting conversation memory.

    Reads  : state["farmer_context"], state["retrieval_result"],
             state["ml_result"], state["chat_history"]
    Writes : groq_prompt, node_logs
    """
    logs = list(state.get("node_logs", []))
    logs.append("[ReasoningNode] Assembling Groq prompt with memory...")

    farmer_context = state.get("farmer_context", {})
    retrieval_result = state.get("retrieval_result", {})
    ml_result = state.get("ml_result", {})
    chat_history = state.get("chat_history", [])

    try:
        from agri_core import create_groq_prompt

        base_prompt = create_groq_prompt(
            farmer_context=farmer_context,
            retrieval_result=retrieval_result,
            ml_result=ml_result,
        )

        # Inject conversation history so the model has context memory
        if chat_history:
            history_text = "\n\n".join(
                f"[{turn['role'].upper()}]: {turn['content']}"
                for turn in chat_history[-10:]   # keep last 10 turns
            )
            memory_section = (
                "\n\n"
                "PREVIOUS CONVERSATION HISTORY (for context memory):\n"
                "=======================================================\n"
                f"{history_text}\n"
                "=======================================================\n"
                "Use the conversation history above to answer follow-up "
                "questions consistently and remember what was discussed earlier.\n"
            )
            # Insert memory before the farmer details section
            prompt = base_prompt.replace(
                "FARMER DETAILS:",
                memory_section + "FARMER DETAILS:",
                1
            )
        else:
            prompt = base_prompt

        logs.append("[ReasoningNode] Prompt assembled.")
        return {
            "groq_prompt": prompt,
            "node_logs": logs,
        }
    except Exception as exc:
        logs.append(f"[ReasoningNode] Error: {exc}")
        return {
            "groq_prompt": "",
            "node_logs": logs,
            "error": f"ReasoningNode failed: {exc}",
        }


# ============================================================
# NODE 5 – REPORT NODE (Groq call + save)
# ============================================================

def report_node(state: AgriState) -> Dict[str, Any]:
    """
    Call Groq LLM and save the report to disk.

    Reads  : state["groq_prompt"], state["retrieval_result"],
             state["ml_result"]
    Writes : final_report, report_saved, node_logs
    """
    logs = list(state.get("node_logs", []))
    logs.append("[ReportNode] Calling Groq agricultural scientist...")

    prompt = state.get("groq_prompt", "")
    if not prompt:
        logs.append("[ReportNode] Empty prompt. Aborting.")
        return {
            "final_report": "Error: empty prompt.",
            "node_logs": logs,
            "error": "Empty groq_prompt in report_node.",
        }

    try:
        from agri_core import call_groq
        report = call_groq(prompt=prompt)   # uses GROQ_PRIMARY_MODEL with auto-fallback
        logs.append("[ReportNode] Groq response received.")
    except Exception as exc:
        logs.append(f"[ReportNode] Groq call failed: {exc}")
        return {
            "final_report": f"Groq error: {exc}",
            "node_logs": logs,
            "error": f"ReportNode Groq failed: {exc}",
        }

    # Save the report to disk
    try:
        output_dir = PROJECT_ROOT / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"agri_report_{timestamp}.txt"
        report_file.write_text(report, encoding="utf-8")
        logs.append(f"[ReportNode] Report saved: {report_file.name}")
        saved_path = str(report_file)
    except Exception as exc:
        logs.append(f"[ReportNode] Could not save report: {exc}")
        saved_path = None

    return {
        "final_report": report,
        "report_saved": saved_path,
        "node_logs": logs,
    }


# ============================================================
# EDGE CONDITION – stop early on error
# ============================================================

def should_stop(state: AgriState) -> str:
    """Route to END immediately if a critical error occurred."""
    if state.get("error"):
        return "end"
    return "continue"


# ============================================================
# BUILD LANGGRAPH GRAPH
# ============================================================

def build_agri_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Returns the compiled graph ready for .invoke() or .stream().
    """
    graph = StateGraph(AgriState)

    # ── Register nodes ───────────────────────────────────────
    graph.add_node("weather_node",    weather_node)
    graph.add_node("retrieval_node",  retrieval_node)
    graph.add_node("ml_node",         ml_node)
    graph.add_node("reasoning_node",  reasoning_node)
    graph.add_node("report_node",     report_node)

    # ── Define edges (linear pipeline with error short-circuit) ──
    graph.add_edge(START, "weather_node")

    # After weather: check for fatal error, else continue
    graph.add_conditional_edges(
        "weather_node",
        should_stop,
        {"end": END, "continue": "retrieval_node"},
    )

    graph.add_conditional_edges(
        "retrieval_node",
        should_stop,
        {"end": END, "continue": "ml_node"},
    )

    graph.add_conditional_edges(
        "ml_node",
        should_stop,
        {"end": END, "continue": "reasoning_node"},
    )

    graph.add_conditional_edges(
        "reasoning_node",
        should_stop,
        {"end": END, "continue": "report_node"},
    )

    graph.add_edge("report_node", END)

    return graph.compile()


# ── Exported compiled graph (lazy compile on first import) ──
agri_graph = build_agri_graph()


# ============================================================
# CONVENIENCE RUNNER
# ============================================================

def run_agri_agent(
    farmer_context: Dict[str, Any],
    farmer_question: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    chat_history: Optional[list] = None,
) -> AgriState:
    """
    High-level function: run the full agent graph for one query.

    Args:
        farmer_context   : Farmer + farm details dict
        farmer_question  : The question string
        latitude         : GPS latitude (optional)
        longitude        : GPS longitude (optional)
        chat_history     : List of {"role", "content"} for memory

    Returns:
        Final AgriState after all nodes have run
    """
    initial_state: AgriState = {
        "farmer_context":  farmer_context,
        "farmer_question": farmer_question,
        "latitude":        latitude,
        "longitude":       longitude,
        "chat_history":    chat_history or [],
        "node_logs":       [],
    }

    result = agri_graph.invoke(initial_state)
    return result


# ============================================================
# CLI TEST (Sprint 1 verification)
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SPRINT 1 — LANGGRAPH GRAPH TEST")
    print("=" * 70)

    # Minimal sample farmer context (no ML files needed for weather + retrieval test)
    sample_farmer = {
        "name": "Test Farmer",
        "state": "Karnataka",
        "district": "Dharwad",
        "crop": "sugarcane",
        "area": 5.0,
        "soil_type": "black",
        "water_source": "borewell",
    }

    print("\n[Graph] Running pipeline…")
    try:
        final = run_agri_agent(
            farmer_context=sample_farmer,
            farmer_question="What crop should I grow this season?",
            latitude=15.3173,
            longitude=75.7139,
            chat_history=[],
        )

        print("\n[Logs]")
        for log in final.get("node_logs", []):
            print(" ", log)

        print("\n[Final Report (first 500 chars)]")
        print(final.get("final_report", "")[:500])

        if final.get("error"):
            print(f"\n[ERROR] {final['error']}")

    except Exception as e:
        traceback.print_exc()
        print(f"\n[FATAL] {e}")
