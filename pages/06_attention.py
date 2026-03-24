import streamlit as st
import numpy as np
from components.state_manager import is_inference_done
from components.ui_helpers import require_inference, next_step_button
from components.html_component import render_component
from components import explanations, tooltips

st.title("Attention")
st.write(
    "**Self-attention** is the core mechanism of transformers. It lets each token look at "
    "every other token and decide which ones are relevant. Below you can explore the attention "
    "patterns learned by DistilGPT-2."
)

# ── The Problem Attention Solves ──────────────────────────────────────────
st.header("Why Attention? The Problem with RNNs")
st.write(
    "Before transformers, language models used **Recurrent Neural Networks (RNNs)** that processed "
    "text one token at a time, passing a hidden state forward like a game of telephone."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Sequential**")
    st.write(
        "Each token depends on the previous one. A 512-token sequence requires 512 sequential steps. "
        "GPUs are built for parallelism — RNNs couldn't use it."
    )
with col2:
    st.markdown("**Vanishing Gradients**")
    st.write(
        "Gradient signal degrades over long sequences. By token 100, the network has essentially "
        "\"forgotten\" token 1. Long-range dependencies couldn't be learned."
    )
with col3:
    st.markdown("**Bottleneck**")
    st.write(
        "All context must compress into one fixed-size vector. A 10,000-word document had to fit "
        "the same bottle as a 10-word sentence. Information was lost."
    )

st.success(
    "**The attention insight:** Instead of compressing context into one vector, let every token "
    "look at every other token directly. This is parallelizable, has no bottleneck, and no "
    "vanishing gradient across positions."
)

require_inference()

tokens = st.session_state["tokens"]
attention_weights = st.session_state["attention_weights"]
num_layers = len(attention_weights)
num_heads = attention_weights[0].shape[0]

# ── Controls ───────────────────────────────────────────────────────────────
st.header("1. Explore Attention Patterns")

col1, col2, col3 = st.columns(3)
with col1:
    layer = st.selectbox(
        "Layer",
        range(num_layers),
        format_func=lambda x: f"Layer {x}" + (" (early)" if x == 0 else " (final)" if x == num_layers - 1 else ""),
        help=tooltips.ATTENTION["layer"],
    )
with col2:
    head_options = ["All Heads"] + list(range(num_heads))
    head_selection = st.selectbox(
        "Head",
        head_options,
        format_func=lambda x: "All Heads (averaged)" if x == "All Heads" else f"Head {x}",
        help=tooltips.ATTENTION["head"],
    )
with col3:
    threshold = st.slider(
        "Min attention weight",
        0.0, 0.5, 0.0, 0.01,
        help=tooltips.ATTENTION["threshold"],
    )

# ── Prepare attention data ─────────────────────────────────────────────────
if head_selection == "All Heads":
    weights = np.mean(attention_weights[layer], axis=0)  # Average across heads
    head_label = "All Heads (averaged)"
else:
    weights = attention_weights[layer][head_selection]
    head_label = f"Head {head_selection}"

# ── View toggle ────────────────────────────────────────────────────────────
view = st.radio("View", ["Heatmap", "Arc Diagram"], horizontal=True)

# ── Visualization ──────────────────────────────────────────────────────────
st.header("2. Attention Visualization")
st.caption(f"Layer {layer}, {head_label}")

if view == "Heatmap":
    render_component("attention_viz", {
        "tokens": [t.strip() if t.strip() else repr(t) for t in tokens],
        "weights": weights.tolist(),
        "threshold": threshold,
        "view": "heatmap",
        "layer": layer,
        "head": str(head_selection),
    }, height=max(400, len(tokens) * 40 + 120))
else:
    render_component("attention_viz", {
        "tokens": [t.strip() if t.strip() else repr(t) for t in tokens],
        "weights": weights.tolist(),
        "threshold": threshold,
        "view": "arcs",
        "layer": layer,
        "head": str(head_selection),
    }, height=350)

# ── Insights ───────────────────────────────────────────────────────────────
st.header("3. What to Look For")
st.info(
    "**Try these experiments:**\n"
    "- Compare **Layer 0** (early) vs **Layer 5** (final) — earlier layers often show local patterns, "
    "later layers show more global relationships.\n"
    "- Look at individual heads — each head learns different patterns.\n"
    "- Increase the threshold slider to see only the strongest attention connections.\n"
    "- Notice how tokens often attend to themselves and to the first token."
)

# ── Learn More ─────────────────────────────────────────────────────────────
st.header("4. Learn More")

with st.expander("How does Attention work?"):
    st.markdown(explanations.WHAT_IS_ATTENTION)

with st.expander("What are Q, K, V?"):
    st.markdown(explanations.QKV_EXPLAINED)

with st.expander("Why multiple heads?"):
    st.markdown(explanations.MULTIPLE_HEADS)

# ── Navigation ─────────────────────────────────────────────────────────────
next_step_button("pages/07_transformer_block.py", "Next: The Transformer Block")
