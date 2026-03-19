"""Short help strings for Streamlit widget tooltips."""

TOKENIZATION = {
    "input_text": "Type any text to see how the model breaks it into tokens.",
    "example_sentences": "Try these pre-written examples to see different tokenization patterns.",
}

ATTENTION = {
    "layer": "DistilGPT-2 has 6 transformer layers (0-5). Earlier layers tend to capture local patterns; later layers capture more abstract relationships.",
    "head": "Each layer has 12 attention heads. Each head learns to focus on different types of relationships.",
    "threshold": "Hide attention connections weaker than this value. Increase to focus on the strongest connections.",
}

OUTPUT = {
    "temperature": "Controls randomness. Low (0.1) = very focused/predictable. High (2.0) = very random/creative. Default is 1.0.",
    "top_n": "How many of the highest-probability tokens to display.",
}

GENERATION = {
    "prompt": "The starting text for the model to continue.",
    "temperature": "Controls randomness of generation. Lower = more predictable.",
    "top_k": "Only consider the top K most likely tokens at each step.",
    "top_p": "Only consider tokens whose cumulative probability reaches this threshold.",
    "max_tokens": "Maximum number of tokens to generate.",
    "strategy": "Greedy always picks the top token. Top-K and Top-P add controlled randomness.",
}
