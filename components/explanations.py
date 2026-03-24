"""Long-form educational explanations for 'Learn more' expanders."""

BYTE_PAIR_ENCODING = """
### How Does Byte-Pair Encoding Work?

BPE starts with individual characters and iteratively merges the most frequent pairs:

1. Start with all characters as individual tokens: `['T', 'h', 'e', ' ', 'c', 'a', 't']`
2. Find the most common adjacent pair (e.g., `'t' + 'h'` appears often)
3. Merge that pair into a new token: `['Th', 'e', ' ', 'c', 'a', 't']`
4. Repeat until you reach the desired vocabulary size (GPT-2 uses 50,257 tokens)

This means common words like "the" become single tokens, while rare words get split into
subword pieces. The model can handle any text, even made-up words, because it can always
fall back to character-level tokens.
"""

WHY_SUBWORDS = """
### Why Not Just Use Whole Words?

Using complete words has two problems:

1. **Vocabulary size explosion**: English has hundreds of thousands of words, plus names,
   technical terms, and typos. The vocabulary would be enormous.
2. **Unknown words**: Any word not in the vocabulary can't be processed at all.

Subword tokenization (like BPE) solves both: a vocabulary of ~50,000 tokens can represent
any text efficiently. Common words are single tokens (fast to process), while rare words
are split into recognizable pieces.
"""

WHAT_IS_ATTENTION = """
### How Does Attention Work?

Self-attention answers the question: "For each token, which other tokens should I pay
attention to?"

For every token, the model computes three vectors:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information should I pass along?"

The attention score between two tokens is the dot product of one token's Query with
another's Key. High scores mean those tokens are relevant to each other. The scores
are normalized with softmax so they sum to 1, then used to create a weighted combination
of the Value vectors.

**The formula:**

**Attention(Q, K, V) = softmax(QK^T / √d_k) · V**

- **QK^T**: Dot product of each Query with every Key — produces a matrix of relevance scores
- **√d_k**: Divide by the square root of the key dimension to prevent scores from growing too large
- **softmax**: Normalize scores to sum to 1 (turns scores into attention weights)
- **· V**: Use those weights to create a weighted combination of Value vectors

This is all matrix multiplication — which GPUs are extremely fast at. That's why attention
can be parallelized while RNNs could not.
"""

QKV_EXPLAINED = """
### What Are Q, K, V?

Think of attention like a search engine:

- **Query** = your search query (what you're looking for)
- **Key** = the title of each document (what each token advertises about itself)
- **Value** = the actual content of the document (the useful information)

The model learns these projections during training. Different attention heads learn
different types of relationships — some might focus on syntactic relationships (subject-verb),
while others capture semantic relationships (pronouns and their antecedents).
"""

MULTIPLE_HEADS = """
### Why Multiple Attention Heads?

DistilGPT-2 uses 12 attention heads per layer. Each head independently learns to focus on
different aspects of the input:

- **Head 1** might learn to connect pronouns to their referents
- **Head 2** might focus on adjacent words (local context)
- **Head 3** might track punctuation and sentence boundaries
- **Head 4** might connect verbs to their subjects

The process for each head: **Project** (Q, K, V through separate weight matrices) →
**Attend** (compute attention scores) → **Concat** (join all head outputs) →
**Project** (final linear layer back to model dimension).

Having multiple heads lets the model capture many different types of relationships
simultaneously. Larger models use more heads — GPT-3 uses 96 heads per layer across
96 layers, each operating in a 128-dimensional subspace.
"""

WHAT_ARE_LOGITS = """
### What Are Logits?

**Logits** are the raw, unnormalized scores that the model produces for every token in
its vocabulary. After processing the input through all transformer layers, the final
hidden state is multiplied by the vocabulary embedding matrix to produce one number
per possible next token.

These numbers can be any real value — positive (the model thinks this token is likely)
or negative (unlikely). They aren't probabilities yet; that's what softmax does.
"""

WHAT_IS_SOFTMAX = """
### What Does Softmax Do?

**Softmax** converts the raw logits into a probability distribution:

1. Take each logit value
2. Raise *e* to that power (makes everything positive)
3. Divide by the sum (makes everything add to 1)

The result: every token gets a probability between 0 and 1, and all probabilities
sum to exactly 1. Tokens with higher logits get higher probabilities, and the
exponential makes the differences more extreme — the highest logit "wins" by a lot.
"""

WHAT_IS_TEMPERATURE = """
### How Does Temperature Work?

**Temperature** controls how "confident" or "creative" the model's predictions are.
Before applying softmax, each logit is divided by the temperature value:

- **Temperature < 1.0** (e.g., 0.3): Divides by a small number, making differences
  between logits larger. The top prediction dominates. Output is more predictable.
- **Temperature = 1.0**: No change. Standard behavior.
- **Temperature > 1.0** (e.g., 2.0): Divides by a large number, flattening the
  differences. All tokens become more equally likely. Output is more random/creative.

At temperature 0 (the limit), the model always picks the highest-probability token (greedy).
"""

AUTOREGRESSIVE_GENERATION = """
### What Is Autoregressive Generation?

Language models generate text **one token at a time**. Here's the process:

1. Feed the prompt into the model: "The cat sat on the"
2. The model outputs probabilities for every possible next token
3. **Sample** a token from this distribution (e.g., "mat")
4. Append the sampled token to the input: "The cat sat on the mat"
5. Feed the extended text back through the model
6. Repeat until a stop condition (max length, end token, etc.)

This is called "autoregressive" because each prediction depends on all previous predictions.
The model can't look ahead — it generates left to right, one token at a time.
"""

SAMPLING_STRATEGIES = """
### Temperature vs Top-K vs Top-P

These three parameters control how the next token is selected:

**Greedy** (Temperature=0): Always pick the most likely token. Deterministic but repetitive.

**Top-K**: Only consider the K most likely tokens, ignore everything else. Set K=1 for
greedy, K=50 for moderate variety.

**Top-P (Nucleus Sampling)**: Only consider the smallest set of tokens whose cumulative
probability exceeds P. At P=0.9, you keep tokens until their probabilities sum to 90%.
This adapts automatically — when the model is confident, fewer tokens are kept.

These can be combined. In practice, Top-P with a moderate temperature (0.7-0.9) tends
to produce the most natural-sounding text.
"""
