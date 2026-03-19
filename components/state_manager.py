import streamlit as st


def init_state():
    """Initialize all session state keys with defaults. Called once per app load."""
    defaults = {
        # Text input
        "input_text": "The quick brown fox jumps",
        # Tokenization
        "tokens": None,
        "token_ids": None,
        # Inference results (cached per input_text)
        "inference_cache_key": None,
        "attention_weights": None,
        "hidden_states": None,
        "logits": None,
        "probabilities": None,
        "top_tokens": None,
        # Generation
        "generation_history": [],
        "generated_text": "",
        "generation_params": {
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.9,
            "max_length": 30,
        },
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def is_text_tokenized() -> bool:
    return st.session_state.get("tokens") is not None


def is_model_loaded() -> bool:
    return True  # Model loads via @st.cache_resource, always available after first call


def is_inference_done() -> bool:
    return st.session_state.get("attention_weights") is not None


def clear_downstream(from_stage: str):
    """Clear session state for stages downstream of a given stage."""
    stages = {
        "text": [
            "tokens", "token_ids",
            "inference_cache_key", "attention_weights", "hidden_states",
            "logits", "probabilities", "top_tokens",
            "generation_history", "generated_text",
        ],
        "inference": [
            "logits", "probabilities", "top_tokens",
            "generation_history", "generated_text",
        ],
        "generation": ["generation_history", "generated_text"],
    }
    keys_to_clear = stages.get(from_stage, [])
    defaults = {"generation_history": []}
    for key in keys_to_clear:
        st.session_state[key] = defaults.get(key, None)
