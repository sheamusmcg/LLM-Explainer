import streamlit as st
import numpy as np
from components.state_manager import is_inference_done
from components.ui_helpers import require_inference, next_step_button
from components.html_component import render_component
from components.inference import apply_temperature
from components.model_utils import load_tokenizer
from components import explanations, tooltips

st.title("Output & Probabilities")
st.write(
    "After processing the input through all transformer layers, the model produces a "
    "**probability distribution** over its entire vocabulary (~50,000 tokens). The token "
    "with the highest probability is the model's best guess for the next word."
)

require_inference()

tokenizer = load_tokenizer()
tokens = st.session_state["tokens"]
logits = st.session_state["logits"]
input_text = st.session_state["input_text"]

# ── Controls ───────────────────────────────────────────────────────────────
st.header("1. Adjust Temperature")

col1, col2 = st.columns([2, 1])
with col1:
    temperature = st.slider(
        "Temperature",
        0.1, 2.0, 1.0, 0.1,
        help=tooltips.OUTPUT["temperature"],
    )
with col2:
    top_n = st.number_input(
        "Show top N tokens",
        10, 100, 30,
        help=tooltips.OUTPUT["top_n"],
    )

# ── Compute probabilities with temperature ─────────────────────────────────
probs = apply_temperature(logits, temperature)
top_indices = np.argsort(probs)[::-1][:top_n]
top_tokens = [
    {"token": tokenizer.decode([int(idx)]), "probability": float(probs[idx])}
    for idx in top_indices
]

# ── Context display ────────────────────────────────────────────────────────
st.header("2. Next Token Prediction")
st.write(f"Given the input: **\"{input_text}\"**")
st.write(f"The model predicts the next token. Top prediction: **\"{top_tokens[0]['token']}\"** "
         f"({top_tokens[0]['probability']*100:.1f}% probability)")

# ── Visualization ──────────────────────────────────────────────────────────
render_component("output_viz", {
    "top_tokens": top_tokens,
    "temperature": temperature,
    "input_text": input_text,
    "context_tokens": [t.strip() if t.strip() else repr(t) for t in tokens],
}, height=max(400, top_n * 22 + 100))

# ── Temperature comparison ─────────────────────────────────────────────────
st.header("3. Temperature Effect")
st.write("See how different temperatures change the probability distribution:")

temps = [0.3, 1.0, 2.0]
cols = st.columns(len(temps))
for col, t in zip(cols, temps):
    with col:
        p = apply_temperature(logits, t)
        top_idx = np.argsort(p)[::-1][:5]
        st.markdown(f"**Temp = {t}**")
        for idx in top_idx:
            tok = tokenizer.decode([int(idx)])
            prob = float(p[idx])
            bar_len = int(prob * 40)
            st.text(f"{'█' * bar_len} {tok!r} {prob*100:.1f}%")

# ── Learn More ─────────────────────────────────────────────────────────────
st.header("4. Learn More")

with st.expander("What are Logits?"):
    st.markdown(explanations.WHAT_ARE_LOGITS)

with st.expander("What does Softmax do?"):
    st.markdown(explanations.WHAT_IS_SOFTMAX)

with st.expander("How does Temperature work?"):
    st.markdown(explanations.WHAT_IS_TEMPERATURE)

# ── Navigation ─────────────────────────────────────────────────────────────
next_step_button("pages/09_generation.py", "Next: Text Generation")
