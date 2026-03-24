import streamlit as st
import numpy as np
from components.state_manager import is_inference_done
from components.ui_helpers import require_inference, next_step_button

st.title("The Transformer Block")
st.write(
    "Now you've seen the individual pieces — embeddings, attention, activation functions. "
    "This page shows how they fit together into the **transformer block**: the repeating unit "
    "that makes up every LLM."
)

require_inference()

tokens = st.session_state["tokens"]
hidden_states = st.session_state["hidden_states"]
attention_weights = st.session_state["attention_weights"]
num_layers = len(attention_weights)

# ── Architecture Diagram ─────────────────────────────────────────────────
st.header("1. The Block")
st.write("Each transformer block repeats the same four steps:")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("### Attend")
    st.write("Multi-head self-attention gathers context from all tokens.")
with col2:
    st.markdown("### Add & Norm")
    st.write("Residual connection (skip) + layer normalization to stabilize training.")
with col3:
    st.markdown("### Feed-Forward")
    st.write("Two linear layers with GELU. Expands to 4× model dimension, then back. Most parameters live here.")
with col4:
    st.markdown("### Add & Norm")
    st.write("Another residual connection + normalization. Output goes to the next layer.")

st.info(
    "DistilGPT-2 stacks **6 of these blocks**. GPT-3 uses 96. Llama 3 8B uses 32. "
    "Same block, different depth — that's the only architectural difference."
)

# ── Residual Connections ─────────────────────────────────────────────────
st.header("2. Why Residual Connections Matter")
st.write(
    "Each sub-layer's output is **added** to its input: `output = input + F(input)`. "
    "This skip connection lets gradients flow directly backward during training, "
    "preventing the vanishing gradient problem that killed deep RNNs."
)

st.latex(r"\text{output} = \text{LayerNorm}(x + \text{Attention}(x))")
st.latex(r"\text{output} = \text{LayerNorm}(x + \text{FFN}(x))")

st.write(
    "Without residual connections, stacking more than ~10 layers causes gradients to shrink "
    "to near-zero. With them, you can stack 96+ layers and still train effectively."
)

# ── Layer-by-Layer Exploration ───────────────────────────────────────────
st.header("3. Watch Representations Evolve")
st.write(
    "As a token passes through each layer, its hidden state is refined. "
    "Early layers capture syntax and local patterns. Later layers capture semantics "
    "and longer-range relationships."
)

# Show how the hidden state magnitude changes across layers
st.write("**Hidden state magnitude per layer** for each token:")

selected_token = st.selectbox(
    "Select a token",
    range(len(tokens)),
    format_func=lambda i: f'Token {i}: "{tokens[i].strip() or repr(tokens[i])}"',
    key="block_token_select",
)

# hidden_states[0] is the embedding output, hidden_states[1] is after layer 0, etc.
layer_labels = ["Embedding"] + [f"Layer {i}" for i in range(num_layers)]
magnitudes = []
for layer_idx in range(num_layers + 1):
    vec = hidden_states[layer_idx][selected_token]
    magnitudes.append(float(np.linalg.norm(vec)))

st.bar_chart(
    {"Layer": layer_labels, "Magnitude": magnitudes},
    x="Layer",
    y="Magnitude",
    height=300,
)
st.caption(
    "The magnitude of the hidden state typically grows through the layers as the model "
    "builds a richer representation. The residual connections contribute to this growth."
)

# ── Cosine similarity across layers ──────────────────────────────────────
st.header("4. How Much Does Each Layer Change?")
st.write(
    "Cosine similarity between a token's representation at consecutive layers. "
    "Low similarity = that layer made a big change. High = minor refinement."
)

similarities = []
for i in range(1, num_layers + 1):
    vec_prev = hidden_states[i - 1][selected_token]
    vec_curr = hidden_states[i][selected_token]
    cos_sim = float(np.dot(vec_prev, vec_curr) / (np.linalg.norm(vec_prev) * np.linalg.norm(vec_curr) + 1e-8))
    similarities.append(cos_sim)

sim_labels = [f"→ Layer {i}" for i in range(num_layers)]
st.bar_chart(
    {"Transition": sim_labels, "Cosine Similarity": similarities},
    x="Transition",
    y="Cosine Similarity",
    height=300,
)
st.caption(
    "If a transition has low cosine similarity, that layer significantly altered "
    "the token's representation — it learned something new about the token's context."
)

# ── Feed-Forward Network ─────────────────────────────────────────────────
st.header("5. The Feed-Forward Network")
st.write(
    "After attention gathers context, the feed-forward network (FFN) processes each token "
    "independently. It's where most of the model's parameters live."
)

st.latex(r"\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2")

st.write(
    "The FFN expands the representation to 4× the model dimension (768 → 3072), applies GELU "
    "activation, then projects back down (3072 → 768). This expansion gives the model room "
    "to compute complex transformations before compressing back."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Input dim", "768")
with col2:
    st.metric("Expanded dim", "3,072")
with col3:
    st.metric("Output dim", "768")

# ── The Full Pipeline ────────────────────────────────────────────────────
st.header("6. Putting It All Together")
st.write("The complete path from text to prediction:")

steps = [
    ("Text", "\"The quick brown fox\""),
    ("Tokenize", "Split into subword tokens"),
    ("Embed + Position", "Each token → 768-dim vector with position info"),
    ("× 6 Transformer Blocks", "Attention → Add & Norm → FFN → Add & Norm"),
    ("Final Hidden State", "Rich contextual representation of each token"),
    ("Linear + Softmax", "Project to vocabulary → probability over ~50K tokens"),
    ("Next Token", "Sample from the distribution → append → repeat"),
]

for i, (step, desc) in enumerate(steps):
    if i > 0:
        st.markdown("↓")
    st.markdown(f"**{step}** — {desc}")

st.info(
    "This is the complete architecture of GPT-2, Claude, Llama, and every other decoder-only LLM. "
    "The only differences between models are the number of layers, heads, and embedding dimensions."
)

# ── Learn More ───────────────────────────────────────────────────────────
st.header("7. Learn More")

with st.expander("What is Layer Normalization?"):
    st.markdown(
        "**Layer normalization** normalizes each token's representation to have mean=0 and std=1. "
        "This prevents activations from growing too large or too small as they pass through many layers.\n\n"
        "Without it, training deep networks becomes unstable — gradients either explode or vanish."
    )

with st.expander("Encoder vs. Decoder — what's the difference?"):
    st.markdown(
        "The original 2017 Transformer had both an **encoder** (bidirectional — each token sees all others) "
        "and a **decoder** (causal — each token only sees previous tokens).\n\n"
        "- **Encoder-only** (BERT): Good for understanding tasks — classification, embeddings, search.\n"
        "- **Decoder-only** (GPT-2, Claude, Llama): Good for generation — each token predicts the next.\n"
        "- **Encoder-decoder** (T5, BART): Good for translation and summarization.\n\n"
        "Every LLM you use for chat is decoder-only. That's what DistilGPT-2 is."
    )

with st.expander("What are scaling laws?"):
    st.markdown(
        "Research has shown that model performance improves predictably as you increase model size, "
        "dataset size, and compute. This is called a **scaling law**.\n\n"
        "Key insight: both model size and training data should scale together. "
        "A well-trained small model often beats a poorly trained large one. "
        "That's why a fine-tuned 7B model can outperform a 70B base model on specific tasks."
    )

# ── Navigation ───────────────────────────────────────────────────────────
next_step_button("pages/08_output.py", "Next: Output & Probabilities")
