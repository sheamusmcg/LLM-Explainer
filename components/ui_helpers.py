"""Reusable UI patterns for the Transformer Explainer app."""

import streamlit as st


def require_tokenized():
    """Show a warning and stop if text has not been tokenized."""
    from components.state_manager import is_text_tokenized
    if not is_text_tokenized():
        st.warning("Please tokenize some text first.")
        st.page_link("pages/02_tokenization.py", label="Go to Tokenization", icon=":material/arrow_back:")
        st.stop()


def require_inference():
    """Show a warning and stop if inference has not been run."""
    from components.state_manager import is_inference_done
    if not is_inference_done():
        st.warning("Please tokenize some text first so the model can run.")
        st.page_link("pages/02_tokenization.py", label="Go to Tokenization", icon=":material/arrow_back:")
        st.stop()


def next_step_button(page_path: str, label: str):
    """Render a navigation link to the next step."""
    st.divider()
    st.page_link(page_path, label=label, icon=":material/arrow_forward:")
