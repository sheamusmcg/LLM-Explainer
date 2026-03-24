import streamlit as st

st.title("Transformer Explainer")
st.write("*An interactive, visual guide to how Large Language Models work — using a real GPT-2 model.*")

st.write(
    "This tool lets you see inside a real transformer model (DistilGPT-2) as it processes "
    "text. You'll start with the fundamentals of neural networks, then work through each "
    "stage of the transformer pipeline — from tokens to generated text. No coding required."
)

# ── Pipeline Overview ──────────────────────────────────────────────────────
st.header("How a Transformer Works")
st.write("Here's the journey your text takes through the model:")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown("### 1. Tokenize")
    st.write("Split text into small pieces called tokens")
with col2:
    st.markdown("### 2. Embed")
    st.write("Convert each token into a vector of numbers with position info")
with col3:
    st.markdown("### 3. Attend")
    st.write("Figure out which tokens relate to each other")
with col4:
    st.markdown("### 4. Transform")
    st.write("Process through multiple layers to build understanding")
with col5:
    st.markdown("### 5. Generate")
    st.write("Predict the most likely next token")

# ── What You'll Learn ─────────────────────────────────────────────────────
st.header("What You'll Learn")
st.write(
    "- How **neurons, weights, and training** work — the building blocks of all neural networks\n"
    "- How **activation functions** (ReLU, GELU, Softmax) add the non-linearity that makes learning possible\n"
    "- How text is **tokenized** into subword pieces using Byte-Pair Encoding\n"
    "- How tokens become **embeddings** — vectors of numbers — and how position is encoded\n"
    "- How **self-attention** lets the model understand relationships between words\n"
    "- How the **transformer block** combines attention, feed-forward networks, and residual connections\n"
    "- How the model produces a **probability distribution** over its vocabulary\n"
    "- How **temperature**, **top-k**, and **top-p** control text generation"
)

# ── Key Terms ─────────────────────────────────────────────────────────────
st.header("Key Terms")

with st.expander("What is a Neural Network?"):
    st.write(
        "A **neural network** is a function that learns to map inputs to outputs by adjusting "
        "millions of numbers called weights. It's made up of layers of simple units (neurons) "
        "that each compute a weighted sum of their inputs and apply an activation function. "
        "Every LLM is a neural network — just a very large, specialized one."
    )

with st.expander("What is a Transformer?"):
    st.write(
        "A **Transformer** is a type of neural network architecture introduced in 2017. "
        "It's the foundation of modern language models like GPT, BERT, and LLaMA. "
        "The key innovation is the **attention mechanism**, which lets the model look at "
        "all parts of the input simultaneously rather than processing it word by word."
    )

with st.expander("What is a Token?"):
    st.write(
        "A **token** is a piece of text that the model works with. It's not always a complete "
        "word — it can be a subword, a single character, or even punctuation. For example, "
        "the word 'unhappiness' might be split into 'un', 'happiness'. This subword approach "
        "lets the model handle any text, even words it hasn't seen before."
    )

with st.expander("What is Attention?"):
    st.write(
        "**Attention** is the mechanism that lets the model figure out which parts of the input "
        "are relevant to each other. When processing the word 'it' in a sentence, the model uses "
        "attention to look back and determine what 'it' refers to. Each attention 'head' can "
        "learn to focus on different types of relationships."
    )

with st.expander("What is GPT-2?"):
    st.write(
        "**GPT-2** (Generative Pre-trained Transformer 2) is a language model released by OpenAI "
        "in 2019. We use **DistilGPT-2**, a smaller distilled version with 6 layers and 82 million "
        "parameters. Despite its small size, it demonstrates all the core concepts of modern LLMs. "
        "It was trained on a large corpus of internet text to predict the next word."
    )

with st.expander("What is a Language Model?"):
    st.write(
        "A **language model** is a system that predicts the next word (or token) given the "
        "previous words. By repeatedly predicting the next token and adding it to the input, "
        "the model can generate coherent text. The quality of the model depends on how well "
        "it captures patterns in language from its training data."
    )

# ── Model Info ─────────────────────────────────────────────────────────────
st.header("About the Model")
st.info(
    "This tool runs **DistilGPT-2** — a 6-layer, 12-head, 82M-parameter transformer. "
    "The model loads automatically on the first page that needs it. All inference happens "
    "on the server, and the visualizations update in real time as you type."
)

# ── Get Started ───────────────────────────────────────────────────────────
st.divider()
st.page_link("pages/02_neural_networks.py", label="Get Started: Neural Network Foundations", icon=":material/arrow_forward:")
