import streamlit as st
import numpy as np
from components.ui_helpers import next_step_button

st.title("Activation Functions")
st.write(
    "Without activation functions, a 100-layer network is mathematically equivalent to a single layer. "
    "Activations introduce **non-linearity** — the ingredient that lets deep networks actually learn "
    "complex patterns."
)

# ── Why They Matter ──────────────────────────────────────────────────────
st.header("1. Why Do We Need Them?")
st.write(
    "Linear operations stack as linear operations: W₃(W₂(W₁x)) = W_combined × x. "
    "Without activation functions, adding more layers adds **zero** learning capacity. "
    "Activations break this by introducing curves, thresholds, and conditional behaviour."
)

# ── Interactive Visualizations ───────────────────────────────────────────
st.header("2. Explore the Functions")
st.write("Adjust the input value and see how each activation transforms it.")

input_val = st.slider("Input value (x)", -5.0, 5.0, 0.0, 0.1)

# Compute activations
relu_out = max(0.0, input_val)

# GELU approximation: x * Φ(x) using tanh approximation
gelu_out = 0.5 * input_val * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (input_val + 0.044715 * input_val**3)))

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### ReLU")
    st.latex(r"f(x) = \max(0, x)")
    st.metric("Output", f"{relu_out:.3f}")
    st.caption("If positive → pass through. If negative → output 0.")
    st.write("Used in: CNNs, early deep learning, simpler networks.")

with col2:
    st.markdown("### GELU")
    st.latex(r"f(x) \approx x \cdot \Phi(x)")
    st.metric("Output", f"{gelu_out:.3f}")
    st.caption("Smooth gate — probabilistically passes or blocks values.")
    st.write("Used in: GPT-2, GPT-3, BERT, and most modern LLMs.")

with col3:
    st.markdown("### Softmax")
    st.write("Converts a vector of raw scores into probabilities that sum to 1.")
    st.write("Used at the **output layer** of every language model to produce next-token probabilities.")
    st.caption("We'll see this in action on the Output & Probabilities page.")

# ── Plot all functions ───────────────────────────────────────────────────
st.header("3. Function Shapes")
st.write("These curves show how each function transforms numbers across a range:")

x = np.linspace(-5, 5, 200)
relu_y = np.maximum(0, x)
gelu_y = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

import streamlit as st

tab1, tab2, tab3 = st.tabs(["ReLU", "GELU", "Side by Side"])

with tab1:
    st.line_chart({"x": x, "ReLU": relu_y}, x="x", y="ReLU", height=300)
    st.caption("Sharp corner at x=0. Flat at 0 for all negative inputs, straight line for positive.")

with tab2:
    st.line_chart({"x": x, "GELU": gelu_y}, x="x", y="GELU", height=300)
    st.caption("Smooth — no sharp corner. Slight dip below 0 near x=0. Approaches ReLU for large x.")

with tab3:
    st.line_chart({"x": x, "ReLU": relu_y, "GELU": gelu_y}, x="x", y=["ReLU", "GELU"], height=300)
    st.caption("GELU is a smoother version of ReLU. LLMs prefer GELU; the difference is subtle but measurable.")

# ── Softmax Demo ─────────────────────────────────────────────────────────
st.header("4. Softmax in Action")
st.write(
    "Softmax converts raw model scores (**logits**) into a probability distribution. "
    "Adjust the logits below to see how the probabilities change."
)

col1, col2, col3 = st.columns(3)
with col1:
    logit_cat = st.slider('"cat" logit', -5.0, 10.0, 3.2, 0.1)
with col2:
    logit_dog = st.slider('"dog" logit', -5.0, 10.0, 1.5, 0.1)
with col3:
    logit_fish = st.slider('"fish" logit', -5.0, 10.0, 0.8, 0.1)

logits = np.array([logit_cat, logit_dog, logit_fish])
exp_logits = np.exp(logits - np.max(logits))
probs = exp_logits / exp_logits.sum()

labels = ["cat", "dog", "fish"]
col_before, col_arrow, col_after = st.columns([2, 1, 2])

with col_before:
    st.markdown("**Raw Logits**")
    for label, logit in zip(labels, logits):
        st.text(f'  "{label}": {logit:.1f}')

with col_arrow:
    st.markdown("")
    st.markdown("")
    st.markdown("### → softmax →")

with col_after:
    st.markdown("**Probabilities (sum = 1)**")
    for label, prob in zip(labels, probs):
        bar = "█" * int(prob * 30)
        st.text(f'  "{label}": {prob:.2%}  {bar}')

# ── Learn More ───────────────────────────────────────────────────────────
st.header("5. Learn More")

with st.expander("Why does GPT-2 use GELU instead of ReLU?"):
    st.markdown(
        "GELU produces slightly better results in transformer models because its smooth shape "
        "helps with gradient flow during training. ReLU has a 'dead neuron' problem — if a neuron "
        "always receives negative input, its gradient is permanently zero and it stops learning. "
        "GELU avoids this because it never fully zeroes out negative values.\n\n"
        "The computational cost difference is negligible at scale."
    )

with st.expander("What does temperature do to softmax?"):
    st.markdown(
        "**Temperature** scales the logits before softmax: `softmax(logits / temperature)`.\n\n"
        "- **Low temperature** (e.g., 0.3): Amplifies differences. The top prediction dominates.\n"
        "- **High temperature** (e.g., 2.0): Flattens differences. All tokens become more equally likely.\n\n"
        "You'll explore this interactively on the Output & Probabilities page."
    )

# ── Navigation ───────────────────────────────────────────────────────────
next_step_button("pages/04_tokenization.py", "Next: Tokenization")
