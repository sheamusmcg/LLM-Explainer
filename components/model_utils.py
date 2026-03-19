"""Model and tokenizer loading with Streamlit caching."""

import streamlit as st


@st.cache_resource(show_spinner="Loading DistilGPT-2 model (first time only)...")
def load_model():
    """Load DistilGPT-2 model. Cached across all sessions and reruns."""
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "distilgpt2",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading tokenizer...")
def load_tokenizer():
    """Load DistilGPT-2 tokenizer. Cached across all sessions and reruns."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    return tokenizer
