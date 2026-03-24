import streamlit as st
from components.model_utils import load_tokenizer, load_model
from components.inference import generate_step
from components.html_component import render_component
from components import explanations, tooltips

st.title("Text Generation")
st.write(
    "Language models generate text **one token at a time**. The model predicts the next token, "
    "appends it to the input, and repeats. Watch the autoregressive process unfold step by step."
)

# ── Load model ─────────────────────────────────────────────────────────────
tokenizer = load_tokenizer()
model = load_model()

# ── Controls ───────────────────────────────────────────────────────────────
st.header("1. Configure Generation")

prompt = st.text_input(
    "Prompt",
    value=st.session_state.get("input_text", "The quick brown fox jumps"),
    help=tooltips.GENERATION["prompt"],
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    strategy = st.radio(
        "Sampling strategy",
        ["Greedy", "Top-K", "Top-P"],
        help=tooltips.GENERATION["strategy"],
    )
with col2:
    temperature = st.slider(
        "Temperature", 0.1, 2.0, 1.0, 0.1,
        help=tooltips.GENERATION["temperature"],
    )
with col3:
    top_k = st.slider(
        "Top-K", 1, 100, 50,
        help=tooltips.GENERATION["top_k"],
        disabled=strategy != "Top-K",
    )
with col4:
    top_p = st.slider(
        "Top-P", 0.1, 1.0, 0.9, 0.05,
        help=tooltips.GENERATION["top_p"],
        disabled=strategy != "Top-P",
    )

max_tokens = st.slider(
    "Max tokens to generate", 1, 50, 20,
    help=tooltips.GENERATION["max_tokens"],
)

# ── Generation ─────────────────────────────────────────────────────────────
st.header("2. Generate")

col_step, col_all, col_reset = st.columns([1, 1, 1])

with col_reset:
    if st.button("Reset", use_container_width=True):
        st.session_state["generation_history"] = []
        st.session_state["generated_text"] = ""
        st.rerun()

# Initialize generation state
if "generation_history" not in st.session_state:
    st.session_state["generation_history"] = []
if "generated_text" not in st.session_state:
    st.session_state["generated_text"] = ""

history = st.session_state["generation_history"]
current_text = prompt + st.session_state["generated_text"]

with col_step:
    if st.button("Generate Next Token", use_container_width=True, type="primary"):
        if len(history) < max_tokens:
            result = generate_step(current_text, tokenizer, model, temperature, top_k, top_p, strategy)
            history.append({
                "step": len(history) + 1,
                "token": result["next_token"],
                "probability": result["probability"],
                "alternatives": result["alternatives"],
            })
            st.session_state["generated_text"] += result["next_token"]
            st.session_state["generation_history"] = history
            st.rerun()

with col_all:
    if st.button("Generate All", use_container_width=True):
        remaining = max_tokens - len(history)
        progress = st.progress(0)
        for i in range(remaining):
            current = prompt + st.session_state["generated_text"]
            result = generate_step(current, tokenizer, model, temperature, top_k, top_p, strategy)
            history.append({
                "step": len(history) + 1,
                "token": result["next_token"],
                "probability": result["probability"],
                "alternatives": result["alternatives"],
            })
            st.session_state["generated_text"] += result["next_token"]
            progress.progress((i + 1) / remaining)
        st.session_state["generation_history"] = history
        st.rerun()

# ── Display ────────────────────────────────────────────────────────────────
if history:
    st.header("3. Generated Output")

    # Token stream visualization
    render_component("generation_viz", {
        "prompt": prompt,
        "steps": history,
        "strategy": strategy,
        "temperature": temperature,
    }, height=250)

    # Full text display
    st.subheader("Full Text")
    st.markdown(f"> {prompt}**{st.session_state['generated_text']}**")

    # Step details
    st.subheader("Step Details")
    st.write("Click on a generated token above for details, or expand a step below:")

    for step in reversed(history[-10:]):  # Show last 10 steps
        with st.expander(f"Step {step['step']}: \"{step['token']}\" ({step['probability']*100:.1f}%)"):
            st.write(f"**Selected token:** `{step['token']}`")
            st.write(f"**Probability:** {step['probability']*100:.2f}%")
            st.write("**Top alternatives:**")
            for alt in step["alternatives"][:5]:
                bar_len = int(alt["probability"] * 40)
                is_selected = alt["token"] == step["token"]
                marker = " ← selected" if is_selected else ""
                st.text(f"  {'█' * bar_len} {alt['token']!r} {alt['probability']*100:.1f}%{marker}")

    st.caption(f"Generated {len(history)} / {max_tokens} tokens")
else:
    st.info("Click **Generate Next Token** to start generating one token at a time, "
            "or **Generate All** to generate all tokens at once.")

# ── Learn More ─────────────────────────────────────────────────────────────
st.header("4. Learn More")

with st.expander("What is Autoregressive Generation?"):
    st.markdown(explanations.AUTOREGRESSIVE_GENERATION)

with st.expander("Temperature vs Top-K vs Top-P"):
    st.markdown(explanations.SAMPLING_STRATEGIES)
