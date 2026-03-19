"""Inference utilities: tokenization, full model run, and generation."""

import numpy as np
import torch


MAX_INPUT_TOKENS = 128  # Limit to keep memory safe on Streamlit Cloud


def tokenize(text, tokenizer):
    """Tokenize text and return token strings and IDs.

    Truncates to MAX_INPUT_TOKENS to stay within memory limits.
    """
    encoded = tokenizer(text, return_offsets_mapping=False,
                        truncation=True, max_length=MAX_INPUT_TOKENS)
    token_ids = encoded["input_ids"]
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    return {"tokens": tokens, "token_ids": token_ids}


def run_inference(text, tokenizer, model):
    """Run full model inference and extract all intermediate outputs.

    Returns dict with attention_weights, hidden_states, logits, probabilities, top_tokens.
    """
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=MAX_INPUT_TOKENS)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    # Attention weights: tuple of (layer,) each [batch, heads, seq, seq]
    attention_weights = {}
    for layer_idx, attn in enumerate(outputs.attentions):
        attention_weights[layer_idx] = attn[0].numpy()  # [heads, seq, seq]

    # Hidden states: tuple of (layer+1,) each [batch, seq, hidden]
    hidden_states = {}
    for layer_idx, hs in enumerate(outputs.hidden_states):
        hidden_states[layer_idx] = hs[0].numpy()  # [seq, hidden]

    # Logits for the last token position
    logits = outputs.logits[0, -1].numpy()  # [vocab_size]

    # Probabilities via softmax
    probabilities = _softmax(logits)

    # Top-50 tokens
    top_indices = np.argsort(probabilities)[::-1][:50]
    top_tokens = [
        {"token": tokenizer.decode([idx]), "token_id": int(idx), "probability": float(probabilities[idx])}
        for idx in top_indices
    ]

    return {
        "attention_weights": attention_weights,
        "hidden_states": hidden_states,
        "logits": logits,
        "probabilities": probabilities,
        "top_tokens": top_tokens,
    }


def apply_temperature(logits, temperature):
    """Apply temperature scaling to logits and return new probabilities + top tokens."""
    scaled = logits / max(temperature, 1e-8)
    probs = _softmax(scaled)
    return probs


def generate_step(text, tokenizer, model, temperature=1.0, top_k=50, top_p=0.9, strategy="Top-K"):
    """Generate a single next token with full intermediate state.

    Returns dict with next_token, probability, alternatives, and the full text.
    """
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=MAX_INPUT_TOKENS)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=False)

    logits = outputs.logits[0, -1]  # [vocab_size]

    # Apply temperature
    scaled_logits = logits / max(temperature, 1e-8)

    # Apply sampling strategy
    if strategy == "Greedy":
        probs = torch.softmax(scaled_logits, dim=-1)
        next_token_id = torch.argmax(probs).item()
    elif strategy == "Top-K":
        probs = _top_k_sampling(scaled_logits, top_k)
        next_token_id = torch.multinomial(probs, 1).item()
    else:  # Top-P
        probs = _top_p_sampling(scaled_logits, top_p)
        next_token_id = torch.multinomial(probs, 1).item()

    probs_np = torch.softmax(scaled_logits, dim=-1).numpy()
    next_token = tokenizer.decode([next_token_id])
    probability = float(probs_np[next_token_id])

    # Top alternatives
    top_indices = np.argsort(probs_np)[::-1][:10]
    alternatives = [
        {"token": tokenizer.decode([int(idx)]), "probability": float(probs_np[idx])}
        for idx in top_indices
    ]

    return {
        "next_token": next_token,
        "next_token_id": next_token_id,
        "probability": probability,
        "alternatives": alternatives,
        "full_text": text + next_token,
    }


def _softmax(x):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def _top_k_sampling(logits, k):
    """Apply top-k filtering then softmax."""
    top_k_vals, _ = torch.topk(logits, k)
    threshold = top_k_vals[-1]
    filtered = torch.where(logits >= threshold, logits, torch.tensor(float('-inf')))
    return torch.softmax(filtered, dim=-1)


def _top_p_sampling(logits, p):
    """Apply top-p (nucleus) filtering then softmax."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[sorted_mask] = float('-inf')

    # Scatter back to original indices
    filtered = torch.zeros_like(logits).fill_(float('-inf'))
    filtered.scatter_(0, sorted_indices, sorted_logits)
    return torch.softmax(filtered, dim=-1)
