# Transformer Explainer

This app was built to demonstrate core aspects of the https://odsc.ai Engineering Accelerator courses

Try it here: https://llm-explainer.streamlit.app/

An an interactive teaching app that lets you look inside a real language model (DistilGPT-2) as it processes text — from raw input to generated output.

Built with Streamlit and Hugging Face Transformers.

## Pages

- **Tokenization** — Type any text and see how the model splits it into subword tokens using Byte-Pair Encoding. Each token is color-coded with its vocabulary ID.
- **Attention** — Explore self-attention with an interactive heatmap or arc diagram. Switch between layers and heads, and adjust a threshold slider to isolate the strongest connections.
- **Output & Probabilities** — See the model's ranked next-token predictions. Adjust temperature to watch the probability distribution shift from confident to random.
- **Text Generation** — Watch autoregressive generation unfold one token at a time. Compare Greedy, Top-K, and Top-P sampling strategies side by side.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Tech Stack

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers (DistilGPT-2)
```
