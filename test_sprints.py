"""
test_sprints.py
===============
Sprint verification tests for the LangGraph + Memory + Streamlit migration.

Run: python test_sprints.py
"""
import sys
import io

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def test_sprint1_state_schema():
    print("\n" + "=" * 60)
    print("TEST: Sprint 1 — AgriState schema import")
    print("=" * 60)
    from agri_graph_state import AgriState
    # Verify all expected keys are documented in the TypedDict
    expected_keys = [
        "farmer_context", "latitude", "longitude", "farmer_question",
        "chat_history", "weather_raw", "weather_summary", "expanded_query",
        "detected_terms", "retrieval_result", "ml_result", "groq_prompt",
        "final_report", "error", "report_saved", "node_logs",
    ]
    annotations = AgriState.__annotations__
    missing = [k for k in expected_keys if k not in annotations]
    assert not missing, f"Missing keys: {missing}"
    print(f"✅ AgriState has all {len(expected_keys)} expected keys.")


def test_sprint1_langgraph_import():
    print("\n" + "=" * 60)
    print("TEST: Sprint 1 — LangGraph graph compile")
    print("=" * 60)
    from agri_graph import agri_graph, run_agri_agent
    # Check the graph has the expected nodes
    node_names = list(agri_graph.nodes.keys())
    print(f"   Graph nodes: {node_names}")
    expected_nodes = ["weather_node", "retrieval_node", "ml_node", "reasoning_node", "report_node"]
    for n in expected_nodes:
        assert n in node_names, f"Missing node: {n}"
    print("✅ LangGraph graph compiled with all 5 nodes.")


def test_sprint2_memory():
    print("\n" + "=" * 60)
    print("TEST: Sprint 2 — ConversationMemory")
    print("=" * 60)
    from agri_memory import ConversationMemory, MemoryStore

    mem = ConversationMemory(window_size=5, session_id="test")
    mem.add_turn("What crop?", "Grow sugarcane.")
    mem.add_turn("What fertilizer?", "Use urea 250 kg/ha.")
    mem.add_turn("Irrigation?", "Drip irrigation weekly.")

    assert mem.turn_count == 6, f"Expected 6 messages, got {mem.turn_count}"
    print(f"✅ Turn count: {mem.turn_count}")

    fmt = mem.format_for_prompt()
    assert "[USER]:" in fmt and "[ASSISTANT]:" in fmt
    print("✅ Format for prompt contains USER and ASSISTANT turns.")

    # Serialization round-trip
    d    = mem.to_dict()
    mem2 = ConversationMemory.from_dict(d)
    assert mem2.turn_count == mem.turn_count
    print("✅ Serialization round-trip passed.")

    # Window eviction
    for i in range(20):
        mem.add_turn(f"Q{i}", f"A{i}")
    assert mem.turn_count <= 10, f"Window eviction failed: {mem.turn_count}"
    print(f"✅ Window eviction: {mem.turn_count} messages (max 10).")

    # MemoryStore
    store = MemoryStore(default_window_size=5)
    m1 = store.get_or_create("session_a")
    m2 = store.get_or_create("session_b")
    assert m1 is not m2
    assert "session_a" in store.active_sessions
    print("✅ MemoryStore manages independent sessions.")


def test_sprint3_streamlit_importable():
    print("\n" + "=" * 60)
    print("TEST: Sprint 3 — streamlit_app.py importable")
    print("=" * 60)
    # We can't actually render Streamlit outside a server,
    # but we can verify the module parses correctly.
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "streamlit_app",
        pathlib.Path(__file__).parent / "streamlit_app.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Don't exec (would call st.set_page_config outside server),
    # just check the file parses as valid Python.
    import ast
    src = (pathlib.Path(__file__).parent / "streamlit_app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)  # raises SyntaxError if invalid
    print("✅ streamlit_app.py parses as valid Python.")
    # Verify key function definitions exist
    func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for fn in ["main", "render_sidebar", "render_map_section", "render_chat_history", "run_pipeline"]:
        assert fn in func_names, f"Missing function: {fn}"
    print(f"✅ All key functions present: {sorted(func_names)}")


def test_sprint3_agri_core_importable():
    print("\n" + "=" * 60)
    print("TEST: agri_core.py importable (no Tkinter dependency)")
    print("=" * 60)
    import ast, pathlib
    src = (pathlib.Path(__file__).parent / "agri_core.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    # Verify no Tkinter imports
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in getattr(node, "names", []):
                assert "tkinter" not in alias.name.lower(), "tkinter found in agri_core.py!"
            if hasattr(node, "module") and node.module:
                assert "tkinter" not in node.module.lower(), "tkinter found in agri_core.py!"
    print("✅ agri_core.py has no Tkinter dependency.")
    # Verify expected classes
    class_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    for cls in ["AgriculturalRetriever", "AgriculturalModels"]:
        assert cls in class_names, f"Missing class: {cls}"
    print(f"✅ Key classes found: {class_names}")


if __name__ == "__main__":
    passed = 0
    failed = 0
    tests = [
        test_sprint1_state_schema,
        test_sprint1_langgraph_import,
        test_sprint2_memory,
        test_sprint3_streamlit_importable,
        test_sprint3_agri_core_importable,
    ]
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n❌ FAILED: {test.__name__}")
            print(f"   {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
