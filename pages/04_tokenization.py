import streamlit as st
from components.model_utils import load_tokenizer, load_model
from components.inference import tokenize, run_inference
from components.state_manager import clear_downstream
from components.html_component import render_component
from components.ui_helpers import next_step_button
from components import explanations, tooltips

st.title("Tokenization")
st.write(
    "Before a language model can process text, it must break it into small pieces called **tokens**. "
    "Type some text below and see how GPT-2's tokenizer splits it up."
)

# ── Load model and tokenizer ──────────────────────────────────────────────
tokenizer = load_tokenizer()
model = load_model()

# ── Text Input ─────────────────────────────────────────────────────────────
st.header("1. Enter Text")

EXAMPLES = {
    "Simple sentence": "The quick brown fox jumps over the lazy dog.",
    "Technical text": "Machine learning algorithms process large datasets efficiently.",
    "Subword splitting": "Unhappiness and misunderstanding are uncomfortable feelings.",
    "Numbers and code": "The price is $42.99 and the function returns array[0].",
    "Repeated tokens": "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.",
}

example = st.selectbox(
    "Try an example",
    ["Custom text"] + list(EXAMPLES.keys()),
    help=tooltips.TOKENIZATION["example_sentences"],
)

if example != "Custom text":
    default_text = EXAMPLES[example]
else:
    default_text = st.session_state.get("input_text", "The quick brown fox jumps")

text = st.text_area(
    "Input text",
    value=default_text,
    height=100,
    help=tooltips.TOKENIZATION["input_text"],
)

if not text.strip():
    st.warning("Please enter some text to tokenize.")
    st.stop()

# ── Tokenize ───────────────────────────────────────────────────────────────
# Check if text changed
if text != st.session_state.get("input_text"):
    st.session_state["input_text"] = text
    clear_downstream("text")

result = tokenize(text, tokenizer)
st.session_state["tokens"] = result["tokens"]
st.session_state["token_ids"] = result["token_ids"]

# ── Visualization ──────────────────────────────────────────────────────────
st.header("2. Token Visualization")

render_component("tokenizer_viz", {
    "text": text,
    "tokens": result["tokens"],
    "token_ids": result["token_ids"],
    "vocab_size": tokenizer.vocab_size,
}, height=280)

# ── Run inference (needed by downstream pages) ─────────────────────────────
if st.session_state.get("inference_cache_key") != text:
    with st.spinner("Running model inference..."):
        inference_result = run_inference(text, tokenizer, model)
        st.session_state["inference_cache_key"] = text
        st.session_state["attention_weights"] = inference_result["attention_weights"]
        st.session_state["hidden_states"] = inference_result["hidden_states"]
        st.session_state["logits"] = inference_result["logits"]
        st.session_state["probabilities"] = inference_result["probabilities"]
        st.session_state["top_tokens"] = inference_result["top_tokens"]

# ── Learn More ─────────────────────────────────────────────────────────────
st.header("3. Learn More")

with st.expander("How does Byte-Pair Encoding work?"):
    st.markdown(explanations.BYTE_PAIR_ENCODING)

with st.expander("Why not just use whole words?"):
    st.markdown(explanations.WHY_SUBWORDS)

# ── Navigation ─────────────────────────────────────────────────────────────
next_step_button("pages/05_embeddings.py", "Next: Embeddings & Position")
