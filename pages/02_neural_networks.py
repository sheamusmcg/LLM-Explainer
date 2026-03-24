import streamlit as st
import numpy as np
from components.ui_helpers import next_step_button

st.title("Neural Network Foundations")
st.write(
    "Before we explore transformers, we need to understand the building block behind every LLM: "
    "the **neural network**. A neural network is a function that learns to map inputs to outputs "
    "by adjusting millions of numbers called **weights**."
)

# ── A Single Neuron ──────────────────────────────────────────────────────
st.header("1. A Single Neuron")
st.write(
    "A neuron takes inputs, multiplies each by a weight, adds a bias, then applies an "
    "activation function. Every neural network — from a simple classifier to a 70B LLM — "
    "is this pattern repeated at scale."
)

st.latex(r"\text{output} = f(x_1 w_1 + x_2 w_2 + x_3 w_3 + b)")
st.caption("f = activation function, w = learned weights, b = learned bias")

# ── Interactive neuron ───────────────────────────────────────────────────
st.header("2. Try It: Single Neuron")
st.write("Adjust the inputs, weights, and bias to see how a neuron computes its output.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Inputs**")
    x1 = st.slider("x₁", -2.0, 2.0, 1.0, 0.1)
    x2 = st.slider("x₂", -2.0, 2.0, 0.5, 0.1)
    x3 = st.slider("x₃", -2.0, 2.0, -0.3, 0.1)
with col2:
    st.markdown("**Weights**")
    w1 = st.slider("w₁", -2.0, 2.0, 0.8, 0.1)
    w2 = st.slider("w₂", -2.0, 2.0, -0.5, 0.1)
    w3 = st.slider("w₃", -2.0, 2.0, 1.2, 0.1)
with col3:
    st.markdown("**Bias & Activation**")
    bias = st.slider("bias (b)", -2.0, 2.0, 0.1, 0.1)
    activation = st.selectbox("Activation", ["ReLU", "None (Linear)"])

# Compute
weighted_sum = x1 * w1 + x2 * w2 + x3 * w3 + bias
if activation == "ReLU":
    output = max(0.0, weighted_sum)
else:
    output = weighted_sum

st.markdown("---")
col_sum, col_out = st.columns(2)
with col_sum:
    st.metric("Weighted Sum (before activation)", f"{weighted_sum:.3f}")
with col_out:
    st.metric("Neuron Output", f"{output:.3f}")

st.info(
    f"**Calculation:** ({x1}×{w1}) + ({x2}×{w2}) + ({x3}×{w3}) + {bias} = {weighted_sum:.3f}"
    + (f" → ReLU → {output:.3f}" if activation == "ReLU" else "")
)

# ── Layers ───────────────────────────────────────────────────────────────
st.header("3. Stacking Neurons into Layers")
st.write(
    "A single neuron isn't very powerful. But stack many neurons into layers, and "
    "connect layers together, and the network can learn complex patterns."
)

st.markdown(
    "- **Input Layer** — Raw data: tokens, pixel values, numbers. No computation, just receives.\n"
    "- **Hidden Layers** — Where learning happens. Each layer finds increasingly abstract patterns.\n"
    "- **Output Layer** — Final prediction: next token probabilities, class labels, etc."
)

st.info(
    "**Key idea:** DistilGPT-2 (the model in this app) has ~82 million weights. "
    "Training adjusts every one of them to make better predictions."
)

# ── How Training Works ───────────────────────────────────────────────────
st.header("4. How a Network Learns")
st.write("Training repeats three steps millions of times:")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### Loss")
    st.write(
        "Measures how wrong the model is. For LLMs: **cross-entropy loss** — "
        "how surprised was the model by the correct next token?"
    )
with col2:
    st.markdown("### Gradient")
    st.write(
        "The direction and amount to nudge each weight to reduce loss. "
        "Computed via **backpropagation** — the chain rule applied backward."
    )
with col3:
    st.markdown("### Update")
    st.write(
        "Move every weight a small step in the direction that reduces loss. "
        "The **learning rate** controls step size."
    )

st.write(
    "One training step: forward pass → compute loss → backprop → update weights. "
    "Repeat ~millions of times. That's it — that's all training is."
)

# ── Learn More ───────────────────────────────────────────────────────────
st.header("5. Learn More")

with st.expander("What is a loss function?"):
    st.markdown(
        "A **loss function** measures the gap between the model's prediction and the correct answer.\n\n"
        "For language models, the standard loss is **cross-entropy**: it measures how surprised the "
        "model was by the actual next token. A loss of 0.0 means perfect prediction. Training typically "
        "starts around 4-5 and aims to get below 1.0.\n\n"
        "The entire goal of training is to minimize this number."
    )

with st.expander("What is gradient descent?"):
    st.markdown(
        "**Gradient descent** is the algorithm that adjusts weights to reduce loss.\n\n"
        "1. Run the input through the model (forward pass)\n"
        "2. Measure the loss\n"
        "3. Calculate the gradient — how much each weight contributed to the error\n"
        "4. Nudge each weight slightly in the direction that reduces loss\n"
        "5. Repeat\n\n"
        "The **learning rate** controls the step size. Too large = unstable training. "
        "Too small = painfully slow."
    )

# ── Navigation ───────────────────────────────────────────────────────────
next_step_button("pages/03_activations.py", "Next: Activation Functions")
