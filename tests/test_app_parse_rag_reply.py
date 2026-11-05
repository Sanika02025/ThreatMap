"""
Tests for parse_rag_reply() in app.py.

Covers the three reply classifications:
  - MAPPED               : T-code(s) found
  - NO_MAPPING           : explicit "No Mapping" or empty string
  - REFUSED_OR_UNPARSEABLE : text with neither a T-code nor "No Mapping"

No Ollama calls, no embeddings — pure string-parsing unit tests.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import parse_rag_reply and the sentinel constants directly from app.py
# without triggering Streamlit's execution model (we import the module-level
# names after patching st so the script-level st.set_page_config() etc. do
# not error out in a non-browser context).
# ---------------------------------------------------------------------------

def _load_app_symbols():
    """
    Import parse_rag_reply, NO_MAPPING, REFUSED from app.py using
    importlib so the test does not depend on the Streamlit runtime.
    We stub out `streamlit` to prevent UI initialisation side-effects.
    """
    # Minimal streamlit stub sufficient to survive the top-level app.py code
    import types

    st_stub = types.ModuleType("streamlit")

    class _NoOp:
        def __call__(self, *a, **kw):
            return self
        def __getattr__(self, name):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def __iter__(self):
            return iter([])

    noop = _NoOp()
    for attr in [
        "set_page_config", "markdown", "cache_resource", "sidebar",
        "title", "caption", "radio", "divider", "write", "success",
        "warning", "error", "info", "toggle", "header", "subheader",
        "columns", "text_area", "text_input", "button", "selectbox",
        "slider", "checkbox", "expander", "dataframe", "metric",
        "plotly_chart", "file_uploader", "download_button", "code",
        "spinner", "tabs", "session_state", "stop",
    ]:
        setattr(st_stub, attr, noop)
    st_stub.session_state = {}

    sys.modules.setdefault("streamlit", st_stub)

    # Also stub plotly to avoid import errors if not installed
    for mod in ["plotly", "plotly.express"]:
        if mod not in sys.modules:
            stub = types.ModuleType(mod)
            stub.bar = lambda *a, **kw: None
            sys.modules[mod] = stub

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    spec = importlib.util.spec_from_file_location("app_module", app_path)
    app_module = importlib.util.module_from_spec(spec)
    # Execute only once; subsequent calls reuse the cached module
    if "app_module" not in sys.modules:
        sys.modules["app_module"] = app_module
        try:
            spec.loader.exec_module(app_module)
        except SystemExit:
            pass
        except Exception:
            pass  # top-level Streamlit rendering errors are expected outside browser

    return app_module


_app = _load_app_symbols()
parse_rag_reply = _app.parse_rag_reply
NO_MAPPING = _app.NO_MAPPING
REFUSED = _app.REFUSED


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseRagReply:
    """Unit tests for parse_rag_reply(status, codes) = parse_rag_reply(raw, candidate_ids)."""

    CANDS = {"T1566.001", "T1059.001", "T1105"}

    # --- MAPPED cases -------------------------------------------------------

    def test_single_code(self):
        status, codes = parse_rag_reply("T1105 : Ingress Tool Transfer", self.CANDS)
        assert status == "MAPPED"
        assert codes == ["T1105"]

    def test_multi_code(self):
        status, codes = parse_rag_reply(
            "T1059.001 : PowerShell\nT1105 : Ingress Tool Transfer", self.CANDS
        )
        assert status == "MAPPED"
        assert codes == ["T1059.001", "T1105"]

    def test_code_with_surrounding_text(self):
        status, codes = parse_rag_reply(
            "Based on the sentence, I would map it to T1566.001 (Spearphishing Attachment).",
            self.CANDS,
        )
        assert status == "MAPPED"
        assert "T1566.001" in codes

    def test_code_takes_priority_over_no_mapping_phrase(self):
        """If the reply contains both a T-code and 'No Mapping', it is still MAPPED."""
        status, codes = parse_rag_reply("No Mapping. Actually T1105.", self.CANDS)
        assert status == "MAPPED"
        assert codes == ["T1105"]

    def test_duplicate_codes_deduplicated(self):
        status, codes = parse_rag_reply("T1105 : X\nT1105 : X again", self.CANDS)
        assert status == "MAPPED"
        assert codes == ["T1105"]

    # --- NO_MAPPING cases ---------------------------------------------------

    def test_explicit_no_mapping_period(self):
        """Verbatim Appendix D output 'No Mapping.' -> NO_MAPPING."""
        status, codes = parse_rag_reply("No Mapping.", self.CANDS)
        assert status == NO_MAPPING
        assert codes == []

    def test_no_mapping_case_insensitive(self):
        status, codes = parse_rag_reply("no mapping", self.CANDS)
        assert status == NO_MAPPING
        assert codes == []

    def test_no_mapping_with_surrounding_text(self):
        status, codes = parse_rag_reply(
            "After reviewing the sentence, no mapping applies.", self.CANDS
        )
        assert status == NO_MAPPING
        assert codes == []

    def test_empty_string(self):
        """Empty reply -> NO_MAPPING (model returned nothing)."""
        status, codes = parse_rag_reply("", self.CANDS)
        assert status == NO_MAPPING
        assert codes == []

    def test_whitespace_only(self):
        status, codes = parse_rag_reply("   \n\t  ", self.CANDS)
        assert status == NO_MAPPING
        assert codes == []

    # --- REFUSED_OR_UNPARSEABLE cases ---------------------------------------

    def test_refusal_safety_message(self):
        """
        The title line 'Threat Report: Operation Silent Harvest' caused:
        "I can't provide information or guidance on illegal or harmful activities,
         including cyber attacks..."
        This must be REFUSED_OR_UNPARSEABLE, not NO_MAPPING.
        """
        refusal = (
            "I can't provide information or guidance on illegal or harmful activities, "
            "including cyber attacks or hacking. If you have concerns about cybersecurity, "
            "I'd be happy to provide general information on best practices for staying safe online."
        )
        status, codes = parse_rag_reply(refusal, self.CANDS)
        assert status == REFUSED
        assert codes == []

    def test_generic_refusal(self):
        status, codes = parse_rag_reply(
            "I'm sorry, I cannot assist with that request.", self.CANDS
        )
        assert status == REFUSED
        assert codes == []

    def test_unrelated_text_no_code_no_no_mapping(self):
        """Random text with neither a T-code nor 'No Mapping'."""
        status, codes = parse_rag_reply("The quick brown fox.", self.CANDS)
        assert status == REFUSED
        assert codes == []
