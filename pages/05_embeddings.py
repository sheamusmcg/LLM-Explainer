import streamlit as st
import numpy as np
import torch
from components.model_utils import load_tokenizer, load_model
from components.state_manager import is_inference_done
from components.ui_helpers import require_inference, next_step_button

st.title("Embeddings & Positional Encoding")
st.write(
    "After tokenization, each token is converted into a **vector of numbers** called an embedding. "
    "The model also adds **positional information** so it knows the order of the tokens. "
    "Without position encoding, \"the dog bit the man\" and \"the man bit the dog\" would look identical."
)

require_inference()

tokenizer = load_tokenizer()
model = load_model()
tokens = st.session_state["tokens"]
token_ids = st.session_state["token_ids"]
hidden_states = st.session_state["hidden_states"]

# ── Token Embeddings ─────────────────────────────────────────────────────
st.header("1. Token Embeddings")
st.write(
    "Each token ID is looked up in an **embedding table** — a giant matrix where each row is "
    "a learned vector representing one token. DistilGPT-2 uses 768-dimensional embeddings, "
    "meaning each token becomes a list of 768 numbers."
)

# Get raw token embeddings from the model
with torch.no_grad():
    input_ids = torch.tensor([token_ids])
    token_embeddings = model.transformer.wte(input_ids)[0].numpy()  # [seq_len, 768]
    position_embeddings = model.transformer.wpe(torch.arange(len(token_ids)).unsqueeze(0))[0].numpy()

combined = token_embeddings + position_embeddings

# Show selected token embedding
selected_idx = st.selectbox(
    "Select a token to inspect",
    range(len(tokens)),
    format_func=lambda i: f'Token {i}: "{tokens[i].strip() or repr(tokens[i])}" (ID: {token_ids[i]})',
)

st.write(f"**Embedding vector** for \"{tokens[selected_idx].strip()}\" — first 20 of 768 dimensions:")

embed_vals = token_embeddings[selected_idx][:20]
cols = st.columns(10)
for i, val in enumerate(embed_vals[:10]):
    with cols[i]:
        color = "green" if val > 0 else "red"
    st.text(f"[{i}] {val:+.3f}")

cols2 = st.columns(10)
for i, val in enumerate(embed_vals[10:20]):
    with cols2[i]:
        st.text(f"[{i+10}] {val:+.3f}")

st.caption(
    f"Full vector has 768 dimensions. These numbers are learned during training — "
    f"similar tokens end up with similar embedding vectors."
)

# ── Positional Encoding ──────────────────────────────────────────────────
st.header("2. Positional Encoding")
st.write(
    "Attention treats its input as a **set** — it has no concept of order. "
    "\"cat sat mat\" and \"mat sat cat\" would look identical without position info.\n\n"
    "The solution: add a **position embedding** to each token embedding. "
    "DistilGPT-2 uses learned position embeddings — one vector per position, trained alongside the model."
)

st.latex(r"\text{input} = \text{token\_embedding} + \text{position\_embedding}")

# Show position embedding values
st.write(f"**Position embedding** for position {selected_idx} — first 20 of 768 dimensions:")
pos_vals = position_embeddings[selected_idx][:20]
for i in range(0, 20, 10):
    cols = st.columns(10)
    for j, val in enumerate(pos_vals[i:i+10]):
        with cols[j]:
            st.text(f"[{i+j}] {val:+.3f}")

# ── Combined Visualization ───────────────────────────────────────────────
st.header("3. Token + Position = Model Input")
st.write("The model's actual input is the sum of both vectors:")

tok_sample = token_embeddings[selected_idx][:8]
pos_sample = position_embeddings[selected_idx][:8]
combined_sample = tok_sample + pos_sample

cols = st.columns(8)
for i in range(8):
    with cols[i]:
        st.markdown(f"**dim {i}**")
        st.text(f"T: {tok_sample[i]:+.2f}")
        st.text(f"P: {pos_sample[i]:+.2f}")
        st.text(f"= {combined_sample[i]:+.2f}")

# ── Similarity between tokens ────────────────────────────────────────────
st.header("4. Embedding Similarity")
st.write(
    "Tokens with similar meanings end up close together in embedding space. "
    "Select two tokens to see how similar their embeddings are (cosine similarity)."
)

if len(tokens) >= 2:
    col1, col2 = st.columns(2)
    with col1:
        tok_a = st.selectbox(
            "Token A",
            range(len(tokens)),
            index=0,
            format_func=lambda i: f'"{tokens[i].strip() or repr(tokens[i])}"',
            key="sim_a",
        )
    with col2:
        tok_b = st.selectbox(
            "Token B",
            range(len(tokens)),
            index=min(1, len(tokens) - 1),
            format_func=lambda i: f'"{tokens[i].strip() or repr(tokens[i])}"',
            key="sim_b",
        )

    vec_a = token_embeddings[tok_a]
    vec_b = token_embeddings[tok_b]
    cosine_sim = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))

    st.metric("Cosine Similarity", f"{cosine_sim:.4f}")
    st.caption("1.0 = identical directions, 0.0 = unrelated, -1.0 = opposite. Same token with itself = 1.0.")
else:
    st.info("Enter at least two tokens on the Tokenization page to compare embeddings.")

# ── Learn More ───────────────────────────────────────────────────────────
st.header("5. Learn More")

with st.expander("Why 768 dimensions?"):
    st.markdown(
        "The embedding dimension is a design choice that trades capacity for cost. "
        "DistilGPT-2 uses 768. GPT-3 uses 12,288. Larger dimensions let the model encode "
        "more nuanced distinctions between tokens, but require more memory and compute.\n\n"
        "The number isn't magic — it's tuned during model development."
    )

with st.expander("Learned vs. sinusoidal position embeddings"):
    st.markdown(
        "The original 2017 Transformer paper used **sinusoidal** position encodings — "
        "fixed mathematical patterns using sin/cos waves. No learned parameters.\n\n"
        "GPT-2 and DistilGPT-2 use **learned** position embeddings instead — one trainable "
        "vector per position. Most modern LLMs (Llama, Mistral) use **RoPE** (Rotary Position "
        "Embedding), which rotates the Q and K vectors based on position.\n\n"
        "All approaches solve the same problem: giving attention a sense of token order."
    )

# ── Navigation ───────────────────────────────────────────────────────────
next_step_button("pages/06_attention.py", "Next: Attention")
