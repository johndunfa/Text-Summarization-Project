# streamlit_app.py — AI Text Summarizer (Streamlit UI)
#
# Run this app (not inside a notebook cell):
#   streamlit run streamlit_app.py
#
# From a Jupyter notebook you can open a terminal and run the same command,
# or use:  !streamlit run streamlit_app.py
#
# The app imports summarization logic from app.py (same folder).

from __future__ import annotations

import traceback

import streamlit as st

# Reuse your existing backend (extractive + abstractive) from app.py
from app import (
    frequency_extractive_summary,
    normalize_whitespace,
    textrank_extractive_summary,
    transformers_abstractive_summary,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Maximum characters accepted (protects memory UI and model limits)
MAX_INPUT_CHARS = 60_000

# Soft warning when text is long but still allowed
WARN_LONG_CHARS = 25_000

# -----------------------------------------------------------------------------
# Page setup & custom styling (minimal, soft palette, subtle motion)
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Injects CSS: spacing, soft background, card shadows, fade-in for results
st.markdown(
    """
    <style>
      /* Page background */
      .stApp {
        background: linear-gradient(165deg, #f8fafc 0%, #f1f5f9 45%, #eef2ff 100%);
      }
      /* Centered main column feel */
      .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
      }
      /* Header */
      .app-header {
        text-align: center;
        margin-bottom: 1.75rem;
        animation: fadeDown 0.55s ease-out;
      }
      .app-header h1 {
        font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
        font-weight: 700;
        font-size: 2.1rem;
        letter-spacing: -0.03em;
        color: #0f172a;
        margin: 0 0 0.35rem 0;
      }
      .app-header p {
        margin: 0;
        color: #64748b;
        font-size: 1.05rem;
      }
      .output-animate {
        animation: fadeUp 0.45s ease-out;
      }
      @keyframes fadeDown {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
      }
      @keyframes fadeUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
      }
      /* Primary buttons */
      .stButton > button[kind="primary"] {
        border-radius: 10px;
        font-weight: 600;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session state (holds input + output so Clear / Copy work cleanly)
# -----------------------------------------------------------------------------

def _init_state() -> None:
    """Initialize session keys once."""
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""
    if "summary_text" not in st.session_state:
        st.session_state.summary_text = ""
    if "last_error" not in st.session_state:
        st.session_state.last_error = ""
    if "last_error_trace" not in st.session_state:
        st.session_state.last_error_trace = ""


_init_state()


# -----------------------------------------------------------------------------
# Summarization wrapper (calls your Python backend)
# -----------------------------------------------------------------------------

def generate_summary(
    raw_text: str,
    mode: str,
    extractive_variant: str,
    abstractive_model: str,
    num_sentences: int,
    max_new_tokens: int,
) -> str:
    """
    mode: "Extractive" | "Abstractive"
    extractive_variant: "Frequency" | "TextRank"
    """
    text = normalize_whitespace(raw_text)
    if not text:
        raise ValueError("Please paste some text before generating a summary.")

    if len(text) > MAX_INPUT_CHARS:
        raise ValueError(
            f"Text is too long ({len(text):,} characters). "
            f"Please shorten to under {MAX_INPUT_CHARS:,} characters "
            "(or split into smaller parts)."
        )

    if mode == "Extractive":
        if extractive_variant == "Frequency":
            return frequency_extractive_summary(text, n_sentences=num_sentences)
        return textrank_extractive_summary(text, n_sentences=num_sentences)

    # Abstractive: map slider roughly to generation length
    min_new = max(8, int(max_new_tokens * 0.25))
    if min_new >= max_new_tokens:
        min_new = max(1, max_new_tokens - 1)

    return transformers_abstractive_summary(
        text,
        model_id=abstractive_model,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new,
    )


# -----------------------------------------------------------------------------
# UI layout
# -----------------------------------------------------------------------------

# --- Header ---
st.markdown(
    """
    <div class="app-header">
      <h1>AI Text Summarizer</h1>
      <p>Turn long text into a short, readable summary — extractive or abstractive.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Two columns: main editor + controls
col_main, col_side = st.columns([1.65, 1.0], gap="large")

with col_main:
    st.markdown("##### Your document")
    user_input = st.text_area(
        "Paste or type your text below",
        height=320,
        placeholder="Paste paragraphs, articles, or notes here…",
        label_visibility="collapsed",
        key="input_text",
    )

with col_side:
    st.markdown("##### Settings")

    summary_mode = st.radio(
        "Summarization type",
        options=["Extractive", "Abstractive"],
        horizontal=True,
        help="Extractive picks important sentences from your text. Abstractive rewrites a new summary with a neural model (slower, needs download).",
    )

    extractive_kind = st.radio(
        "Extractive method",
        options=["TextRank (recommended)", "Frequency-based"],
        disabled=(summary_mode != "Extractive"),
    )

    model_choice = st.selectbox(
        "Abstractive model",
        options=["t5-small (lighter)", "facebook/bart-large-cnn (heavier, often better)"],
        disabled=(summary_mode != "Abstractive"),
    )
    model_id = "t5-small" if model_choice.startswith("t5") else "facebook/bart-large-cnn"

    num_sentences = st.slider(
        "Summary length (sentences, extractive)",
        min_value=1,
        max_value=6,
        value=3,
        disabled=(summary_mode != "Extractive"),
    )

    max_tokens = st.slider(
        "Max output length (abstractive)",
        min_value=40,
        max_value=200,
        value=90,
        step=10,
        disabled=(summary_mode != "Abstractive"),
    )

# Long-text hint
if user_input and len(user_input) > WARN_LONG_CHARS:
    st.info(
        f"Long input ({len(user_input):,} characters). Summarization may take longer. "
        f"Hard limit: {MAX_INPUT_CHARS:,} characters."
    )

# Action row
btn_cols = st.columns([1, 1, 2], gap="medium")
with btn_cols[0]:
    generate_clicked = st.button("Generate summary", type="primary", use_container_width=True)
with btn_cols[1]:
    clear_clicked = st.button("Clear all", use_container_width=True)

# Clear resets the text area (via session state key) and summary
if clear_clicked:
    st.session_state.input_text = ""
    st.session_state.summary_text = ""
    st.session_state.last_error = ""
    st.session_state.last_error_trace = ""
    st.rerun()

# Generate
if generate_clicked:
    st.session_state.last_error = ""
    st.session_state.last_error_trace = ""
    st.session_state.summary_text = ""

    if not (user_input or "").strip():
        st.session_state.last_error = "Please enter some text first."
    elif len(user_input) > MAX_INPUT_CHARS:
        st.session_state.last_error = (
            f"Input is too long ({len(user_input):,} characters). "
            f"Maximum is {MAX_INPUT_CHARS:,}."
        )
    else:
        try:
            variant = (
                "Frequency"
                if extractive_kind.startswith("Frequency")
                else "TextRank"
            )
            with st.spinner("Generating summary… This may take a moment for abstractive mode."):
                result = generate_summary(
                    user_input,
                    summary_mode,
                    variant,
                    model_id,
                    num_sentences=num_sentences,
                    max_new_tokens=max_tokens,
                )
            st.session_state.summary_text = result or "(No summary produced — try more text.)"
        except Exception as e:
            # Keep the main message short; full trace helps debugging in the expander.
            st.session_state.last_error = f"{type(e).__name__}: {e}"
            st.session_state.last_error_trace = traceback.format_exc()

if st.session_state.last_error:
    st.error(st.session_state.last_error)
    if st.session_state.get("last_error_trace"):
        with st.expander("Technical details (for debugging)"):
            st.code(st.session_state.last_error_trace, language="text")

# --- Output ---
st.markdown("---")
st.markdown("##### Summary")

if st.session_state.summary_text:
    # `disabled=True` keeps the summary read-only; users can still select text to copy.
    st.text_area(
        "Summary output",
        value=st.session_state.summary_text,
        height=220,
        label_visibility="collapsed",
        disabled=True,
    )

    copy_cols = st.columns([1, 3])
    with copy_cols[0]:
        if st.button("Copy to clipboard", key="btn_copy_summary", use_container_width=True):
            try:
                import pyperclip

                pyperclip.copy(st.session_state.summary_text)
                st.toast("Copied to clipboard", icon="✅")
            except Exception:
                st.warning(
                    "Automatic copy failed. Select the summary text and press Ctrl+C (Cmd+C on Mac)."
                )
else:
    st.caption("Your summary will appear here after you click **Generate summary**.")
