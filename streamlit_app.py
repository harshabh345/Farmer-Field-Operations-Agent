"""
streamlit_app.py
================
Streamlit Web Interface for the Farmer Field Operations Agent.

Features:
  - Dark-themed premium UI with agricultural aesthetics
  - Interactive map (folium + streamlit-folium) for GPS selection
  - Sidebar farmer profile form (all fields)
  - Chat-style conversation interface with memory
  - Real-time node-by-node progress display
  - LangGraph-powered pipeline under the hood

Run with:
    python -m streamlit run streamlit_app.py
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title="Farmer Field Operations Agent",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inline CSS — dark agricultural theme ─────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #0d2137 60%, #071a07 100%);
    color: #e8f5e9;
}

/* Hide default Streamlit header chrome */
header[data-testid="stHeader"] {
    background: rgba(10,22,40,0.95) !important;
    border-bottom: 1px solid #1a3a1a;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071020 0%, #071a07 100%);
    border-right: 1px solid #1a3a1a;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] .stMarkdown {
    color: #b2dfdb !important;
}

/* Main header banner */
.main-header {
    background: linear-gradient(135deg, #1b5e20 0%, #0d47a1 100%);
    padding: 1.8rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}
.main-header h1 {
    color: #e8f5e9;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 0 2px 8px rgba(0,0,0,0.5);
}
.main-header p {
    color: #a5d6a7;
    font-size: 0.98rem;
    margin-top: 0.5rem;
    margin-bottom: 0;
}

/* Section titles */
.section-title {
    color: #69f0ae;
    font-size: 1.05rem;
    font-weight: 600;
    border-bottom: 2px solid rgba(105,240,174,0.25);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
}

/* Info / welcome card */
.info-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(105,240,174,0.15);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    color: #cfd8dc;
    line-height: 1.7;
}
.info-card ol { margin: 0.5rem 0 0.5rem 1.2rem; }
.info-card li { margin-bottom: 0.3rem; }

/* Chat bubbles */
.chat-user {
    background: linear-gradient(135deg, #1a3a0a, #0d3318);
    border: 1px solid #2e7d32;
    border-radius: 18px 18px 4px 18px;
    padding: 0.9rem 1.2rem;
    margin: 0.6rem 0;
    margin-left: 12%;
    color: #e8f5e9;
    font-size: 0.93rem;
    box-shadow: 0 4px 16px rgba(46,125,50,0.25);
}
.chat-assistant-header {
    color: #69f0ae;
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 0.4rem;
}
.chat-assistant-wrapper {
    background: linear-gradient(135deg, #0a1a2e, #0d1f0d);
    border: 1px solid #1565c0;
    border-radius: 18px 18px 18px 4px;
    padding: 1rem 1.3rem;
    margin: 0.6rem 0;
    margin-right: 8%;
    color: #e3f2fd;
    font-size: 0.91rem;
    line-height: 1.75;
    box-shadow: 0 4px 16px rgba(21,101,192,0.2);
}
.chat-meta {
    color: #546e7a;
    font-size: 0.72rem;
    margin-top: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Pipeline log terminal */
.node-log {
    background: #040c18;
    border: 1px solid #1a2744;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #4fc3f7;
    max-height: 180px;
    overflow-y: auto;
    line-height: 1.6;
}

/* GPS badge */
.gps-badge {
    background: linear-gradient(135deg, #1b5e20, #0d47a1);
    color: #e8f5e9;
    padding: 0.45rem 1rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-family: 'JetBrains Mono', monospace;
    display: inline-block;
    margin: 0.5rem 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
}

/* Preset question buttons override */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1b5e20cc, #0d47a1cc) !important;
    color: #e8f5e9 !important;
    border: 1px solid rgba(105,240,174,0.25) !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    transition: all 0.25s ease !important;
    padding: 0.5rem 1rem !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: #69f0ae !important;
    box-shadow: 0 4px 16px rgba(105,240,174,0.2) !important;
    transform: translateY(-1px) !important;
}

/* Form submit buttons */
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #2e7d32, #1565c0) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.5rem !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35) !important;
    transition: all 0.25s ease !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    box-shadow: 0 6px 24px rgba(46,125,50,0.45) !important;
    transform: translateY(-2px) !important;
}

/* Text inputs */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #e8f5e9 !important;
    font-size: 0.9rem !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #e8f5e9 !important;
}

/* Map container */
iframe {
    border-radius: 10px;
    border: 1px solid rgba(105,240,174,0.2);
}

/* Spinner */
.stSpinner > div { border-top-color: #69f0ae !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #071020; }
::-webkit-scrollbar-thumb { background: #1b5e20; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2e7d32; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 8px 8px 0 0;
}
.stTabs [data-baseweb="tab"] { color: #90a4ae !important; }
.stTabs [aria-selected="true"] { color: #69f0ae !important; }

/* Expander */
.streamlit-expanderHeader { color: #90caf9 !important; }

/* Alerts */
div[data-testid="stAlert"] { border-radius: 8px !important; }

/* Footer */
.footer-bar {
    text-align: center;
    color: #455a64;
    font-size: 0.78rem;
    padding: 1rem 0;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Import after page config ──────────────────────────────────
from agri_memory import get_streamlit_memory, memory_to_chat_history


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state():
    defaults = {
        "latitude":         None,
        "longitude":        None,
        "chat_messages":    [],
        "farmer_context":   {},
        "profile_saved":    False,
        "pipeline_running": False,
        "node_logs":        [],
        "turn_count":       0,
        "last_report":      "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================
# SIDEBAR — FARMER PROFILE FORM
# ============================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🌾 Farmer Profile")
        st.markdown("---")

        with st.form("farmer_form", clear_on_submit=False):
            st.markdown("**👤 Personal Details**")
            name      = st.text_input("Farmer Name *",        placeholder="e.g. Ramesh Kumar")
            phone     = st.text_input("Phone Number",          placeholder="Optional")
            farm_name = st.text_input("Farm Name *",          placeholder="e.g. Green Valley Farm")

            st.markdown("**📍 Location**")
            state     = st.selectbox("State *", [
                "", "Karnataka", "Maharashtra", "Madhya Pradesh", "Gujarat",
                "Telangana", "Andhra Pradesh", "Uttar Pradesh", "Punjab",
                "Haryana", "Tamil Nadu", "Rajasthan", "Bihar", "West Bengal", "Other",
            ])
            district  = st.text_input("District *",           placeholder="e.g. Dharwad")
            taluk     = st.text_input("Taluk / Tehsil",        placeholder="Optional")
            village   = st.text_input("Village",               placeholder="Optional")

            st.markdown("**🌱 Farm Details**")
            land_area = st.number_input("Total Farm Area (acres) *", min_value=0.1, value=5.0, step=0.5)
            plot_area = st.number_input("Plot Area (acres) *",       min_value=0.1, value=2.0, step=0.5)
            crop      = st.text_input("Current / Planned Crop",      placeholder="e.g. sugarcane, rice")
            season    = st.selectbox("Season", ["", "Kharif", "Rabi", "Zaid", "Annual", "Summer"])

            st.markdown("**🪨 Soil**")
            soil_type  = st.selectbox("Soil Type", ["", "Black", "Red", "Sandy", "Clay", "Loam", "Mixed"])
            soil_ph    = st.text_input("Soil pH",              placeholder="e.g. 6.5")
            nitrogen   = st.text_input("Nitrogen (kg/ha)",     placeholder="Optional")
            phosphorus = st.text_input("Phosphorus (kg/ha)",   placeholder="Optional")
            potassium  = st.text_input("Potassium (kg/ha)",    placeholder="Optional")

            st.markdown("**💧 Water & Irrigation**")
            water_avail = st.selectbox("Water Availability",  ["", "Low", "Medium", "High"])
            water_src   = st.selectbox("Water Source", [
                "", "Borewell", "Canal", "Rainfed", "Tank", "River", "Other"
            ])
            irr_method  = st.selectbox("Irrigation Method", [
                "", "Drip", "Sprinkler", "Flood", "Rainfed", "Furrow", "Other"
            ])

            st.markdown("**📊 History**")
            prev_crops = st.text_input("Previous Crops",       placeholder="e.g. wheat, cotton")
            prev_yield = st.text_input("Previous Yield",       placeholder="e.g. 3.5 t/ha")
            prev_probs = st.text_area("Previous Problems",     placeholder="Pests, drought…", height=55)

            st.markdown("**🌡️ Climate Patterns**")
            rain_pat = st.selectbox("Rainfall",    ["", "Low", "Medium", "High"])
            temp_pat = st.selectbox("Temperature", ["", "Low", "Medium", "High"])
            hum_pat  = st.selectbox("Humidity",    ["", "Low", "Medium", "High"])

            submitted = st.form_submit_button("💾 Save Profile", use_container_width=True)

        if submitted:
            if not name.strip() or not state or not district.strip() or not farm_name.strip():
                st.error("Please fill in all required (*) fields.")
            else:
                st.session_state["farmer_context"] = {
                    "name": name, "phone": phone, "farm_name": farm_name,
                    "state": state, "district": district, "taluk": taluk,
                    "village": village, "land_area_acres": land_area,
                    "plot_area_acres": plot_area, "crop": crop, "season": season,
                    "soil_type": soil_type, "soil_ph": soil_ph,
                    "nitrogen": nitrogen, "phosphorus": phosphorus,
                    "potassium": potassium, "water_availability": water_avail,
                    "water_source": water_src, "irrigation_method": irr_method,
                    "previous_crops": prev_crops, "previous_yield": prev_yield,
                    "previous_problems": prev_probs, "rainfall_pattern": rain_pat,
                    "temperature_pattern": temp_pat, "humidity_pattern": hum_pat,
                }
                st.session_state["profile_saved"] = True
                st.success("✅ Profile saved!")

        # Memory controls
        st.markdown("---")
        st.markdown("**🧠 Conversation Memory**")
        mem = get_streamlit_memory(st.session_state)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.caption(f"Messages: **{mem.turn_count}**")
        with col_m2:
            if st.button("🗑️ Clear", use_container_width=True, key="clear_memory"):
                mem.clear()
                st.session_state["chat_messages"] = []
                st.session_state["turn_count"] = 0
                st.rerun()

        # Profile status
        if st.session_state["profile_saved"]:
            ctx = st.session_state.get("farmer_context", {})
            st.success(f"✅ {ctx.get('name', 'Farmer')} — {ctx.get('state', '')}")
        else:
            st.warning("⚠️ Fill profile & click Save")


# ============================================================
# MAP SECTION
# ============================================================

def render_map_section():
    st.markdown('<div class="section-title">📍 Select Farm Location</div>', unsafe_allow_html=True)

    try:
        import folium
        from streamlit_folium import st_folium

        center_lat = st.session_state.get("latitude") or 20.5937
        center_lon = st.session_state.get("longitude") or 78.9629
        zoom = 12 if st.session_state.get("latitude") else 5

        # Use standard OpenStreetMap (100% free, no API keys, fully reliable)
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles="OpenStreetMap",
            prefer_canvas=True,
        )

        # Place marker if location already selected
        if st.session_state["latitude"] and st.session_state["longitude"]:
            folium.Marker(
                location=[st.session_state["latitude"], st.session_state["longitude"]],
                popup=folium.Popup("🌾 Your Farm", max_width=150),
                tooltip="Your Farm Location",
                icon=folium.Icon(color="green", icon="home", prefix="glyphicon"),
            ).add_to(m)

            # Draw a circle around the farm
            folium.Circle(
                location=[st.session_state["latitude"], st.session_state["longitude"]],
                radius=500,
                color="#69f0ae",
                fill=True,
                fill_opacity=0.1,
                weight=2,
            ).add_to(m)

        folium.LayerControl(collapsed=True).add_to(m)

        # Render map
        map_result = st_folium(
            m,
            width="100%",
            height=380,
            returned_objects=["last_clicked"],
            key="farm_map",
        )

        # Capture click
        if map_result and map_result.get("last_clicked"):
            clicked = map_result["last_clicked"]
            new_lat = round(clicked["lat"], 7)
            new_lon = round(clicked["lng"], 7)
            if new_lat != st.session_state.get("latitude") or new_lon != st.session_state.get("longitude"):
                st.session_state["latitude"]  = new_lat
                st.session_state["longitude"] = new_lon
                st.rerun()

    except ImportError:
        st.warning("Install folium and streamlit-folium for interactive map.")
        st.code("pip install folium streamlit-folium", language="bash")
        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input("Latitude",  value=20.5937, format="%.6f", key="manual_lat")
        with c2:
            lon = st.number_input("Longitude", value=78.9629, format="%.6f", key="manual_lon")
        if st.button("📍 Set Location", key="set_loc_btn"):
            st.session_state["latitude"]  = lat
            st.session_state["longitude"] = lon
            st.rerun()

    # GPS badge
    if st.session_state.get("latitude") and st.session_state.get("longitude"):
        st.markdown(
            f'<span class="gps-badge">🛰️ '
            f'{st.session_state["latitude"]:.5f}°N, '
            f'{st.session_state["longitude"]:.5f}°E</span>',
            unsafe_allow_html=True,
        )
    else:
        st.info("👆 Click anywhere on the map to pin your farm location.")


# ============================================================
# CHAT DISPLAY
# ============================================================

def render_chat_history():
    msgs = st.session_state.get("chat_messages", [])
    if not msgs:
        return
    for msg in msgs:
        role      = msg["role"]
        content   = msg["content"]
        timestamp = msg.get("timestamp", "")
        if role == "user":
            st.markdown(
                f'<div class="chat-user">'
                f'🧑‍🌾 <strong>Farmer</strong><br/>{content}'
                f'<div class="chat-meta">{timestamp}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-assistant-wrapper">'
                f'<div class="chat-assistant-header">🤖 Agricultural AI Scientist'
                f'<span class="chat-meta" style="float:right">{timestamp}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(content)
            st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PIPELINE RUNNER
# ============================================================

def run_pipeline(question: str) -> str:
    """Run the full LangGraph pipeline. Returns the final report."""
    from agri_graph import run_agri_agent

    mem          = get_streamlit_memory(st.session_state)
    chat_history = memory_to_chat_history(mem)
    farmer_ctx   = dict(st.session_state.get("farmer_context", {}))
    farmer_ctx["latitude"]       = st.session_state.get("latitude")
    farmer_ctx["longitude"]      = st.session_state.get("longitude")
    farmer_ctx["farmer_question"] = question

    # Add user message to display history
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state["chat_messages"].append({"role": "user", "content": question, "timestamp": ts})
    mem.add_user_turn(question)

    # Run the graph with live progress
    log_placeholder = st.empty()

    def show_log(logs):
        html = "<br>".join(logs) if logs else "⏳ Initializing…"
        log_placeholder.markdown(f'<div class="node-log">{html}</div>', unsafe_allow_html=True)

    show_log(["⏳ Starting LangGraph pipeline…"])

    try:
        result = run_agri_agent(
            farmer_context=farmer_ctx,
            farmer_question=question,
            latitude=st.session_state.get("latitude"),
            longitude=st.session_state.get("longitude"),
            chat_history=chat_history,
        )
        show_log(result.get("node_logs", []))
        time.sleep(0.5)
        log_placeholder.empty()
    except Exception as e:
        log_placeholder.empty()
        st.error(f"Pipeline error: {e}")
        st.code(traceback.format_exc())
        return ""

    report = result.get("final_report") or f"Error: {result.get('error', 'No report generated.')}"

    # Save to memory and display history
    ts = datetime.now().strftime("%H:%M:%S")
    mem.add_assistant_turn(report)
    st.session_state["chat_messages"].append({"role": "assistant", "content": report, "timestamp": ts})
    st.session_state["turn_count"] += 1
    st.session_state["last_report"] = report

    # Node logs expander (transparency)
    if result.get("node_logs"):
        with st.expander("🔍 Pipeline Node Logs", expanded=False):
            for log in result["node_logs"]:
                st.caption(log)

    # Report saved notification
    if result.get("report_saved"):
        st.success(f"💾 Report saved to `outputs/`")

    return report


# ============================================================
# MAIN APP
# ============================================================

def main():
    init_session_state()

    # ── Header ───────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🌾 Farmer Field Operations Agent</h1>
        <p>Powered by LangGraph &nbsp;·&nbsp; Hybrid RAG &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; Groq LLM &nbsp;·&nbsp; Context Memory</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────
    render_sidebar()

    # ── Top: Status bar ──────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        profile_ok = "✅ Profile saved" if st.session_state["profile_saved"] else "⚠️ Profile not saved"
        st.caption(profile_ok)
    with s2:
        gps_ok = (f"📍 GPS: {st.session_state['latitude']:.4f}, {st.session_state['longitude']:.4f}"
                  if st.session_state.get("latitude") else "📍 No GPS selected")
        st.caption(gps_ok)
    with s3:
        mem = get_streamlit_memory(st.session_state)
        st.caption(f"🧠 Memory: {mem.turn_count} messages")
    with s4:
        st.caption(f"💬 Turns: {st.session_state['turn_count']}")

    st.markdown("---")

    # ── Main layout: 2 columns ───────────────────────────────
    left_col, right_col = st.columns([1.1, 1.4], gap="large")

    # ── LEFT: Map + Quick Questions ─────────────────────────
    with left_col:
        render_map_section()

        st.markdown("---")
        st.markdown('<div class="section-title">💡 Quick Questions</div>', unsafe_allow_html=True)

        preset_qs = [
            "What crop should I grow this season?",
            "What fertilizer schedule should I follow?",
            "How should I manage irrigation this week?",
            "What pests or diseases should I watch for?",
            "What is my expected yield this season?",
        ]
        for q in preset_qs:
            if st.button(q, key=f"q_{q[:25]}", use_container_width=True):
                if not st.session_state["profile_saved"]:
                    st.warning("⚠️ Save your farmer profile first (sidebar).")
                elif not st.session_state.get("latitude"):
                    st.warning("⚠️ Select your farm location on the map.")
                else:
                    with st.spinner("🌾 Agricultural AI thinking…"):
                        run_pipeline(q)
                    st.rerun()

    # ── RIGHT: Conversation ──────────────────────────────────
    with right_col:
        st.markdown('<div class="section-title">💬 Conversation with Agricultural AI</div>', unsafe_allow_html=True)

        # Chat history
        render_chat_history()

        # Welcome card (shown only when no messages yet)
        if not st.session_state["chat_messages"]:
            st.markdown("""
            <div class="info-card">
                <strong>👋 Welcome to the Farmer Field Operations Agent!</strong><br/><br/>
                <strong>To get started:</strong>
                <ol>
                    <li>📝 Fill in your <strong>Farmer Profile</strong> in the left sidebar and click <em>Save Profile</em></li>
                    <li>📍 Click on the <strong>map</strong> to pin your exact farm location (GPS)</li>
                    <li>❓ Type your question below or pick a <strong>Quick Question</strong></li>
                </ol>
                The AI analyses your soil, live weather, crop history, and ICAR/KCC scientific evidence
                to give you evidence-based agricultural advice in real time.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Question input form
        with st.form("ask_form", clear_on_submit=True):
            question = st.text_area(
                "Your question:",
                placeholder="e.g. What crop should I grow given my black soil and the current monsoon weather?",
                height=95,
                label_visibility="collapsed",
            )
            ask_btn = st.form_submit_button("🌾 Ask the Agent", use_container_width=True)

        if ask_btn:
            if not question.strip():
                st.warning("Please type a question.")
            elif not st.session_state["profile_saved"]:
                st.warning("⚠️ Save your farmer profile first (sidebar).")
            elif not st.session_state.get("latitude"):
                st.warning("⚠️ Select your farm location on the map first.")
            else:
                with st.spinner("🌾 Agricultural AI is analysing your farm situation…"):
                    run_pipeline(question.strip())
                st.rerun()

    # ── Footer ────────────────────────────────────────────────
    st.markdown("""
    <div class="footer-bar">
        🌾 Farmer Field Operations Agent &nbsp;|&nbsp;
        LangGraph + Hybrid RAG + XGBoost + Groq LLM &nbsp;|&nbsp;
        Evidence: ICAR / KCC / Open-Meteo &nbsp;|&nbsp;
        <em>Always verify with your local KVK or agriculture officer.</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
