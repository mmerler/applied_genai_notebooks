import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Module 9: Practical - Transformer Architecture
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import re

    torch.manual_seed(42)
    np.random.seed(42)

    # Check which GPU is available
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    print(f"Using device: {device}")
    return DataLoader, Dataset, device, nn, np, plt, re, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    We start with the same data preparation steps as in Module 5.
    """)
    return


@app.cell
def _(re):
    # Load and preprocess Count of Monte Cristo
    url = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"

    import requests
    text = requests.get(url).text

    # Keep only the main body (remove header/footer)
    start_idx = text.find("Chapter 1")
    end_idx = text.rfind("Chapter 5") # text.rfind("End of the Project Gutenberg")
    text = text[start_idx:end_idx]

    # Pre-processing
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()

    # Tokenization
    tokens = text.split()

    # Vocabulary construction
    from collections import Counter
    counter = Counter(tokens)

    # We'll assign indices 0 and 1 to special tokens "<PAD>" and "<UNK>", the rest of the indeces
    # are based on the frequency of the words.
    vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common(9998))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    inv_vocab = {idx: word for word, idx in vocab.items()}

    # Encode tokens
    encoded = [vocab.get(word, vocab["<UNK>"]) for word in tokens]
    return encoded, inv_vocab


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Since we are training the model to predict the next word in a sequence, we will construct our training set features based on 30 word sequences from the text. The corresponding labels are the sequences shifted by one word.

    We also hold out the last 10% of the text as a **validation set**. It is a contiguous block, so no validation window overlaps with a training window. We will use it to measure the model's **perplexity** on text it has never seen.
    """)
    return


@app.cell
def _(DataLoader, Dataset, encoded, torch):
    # Create sequences
    SEQ_LEN = 30
    class TextDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data) - SEQ_LEN

        def __getitem__(self, idx):
            return (torch.tensor(self.data[idx:idx+SEQ_LEN]),
                    torch.tensor(self.data[idx+1:idx+SEQ_LEN+1]))

    # Hold out the last 10% of the text for validation
    split = int(0.9 * len(encoded))
    train_datasets = TextDataset(encoded[:split])
    val_datasets = TextDataset(encoded[split:])
    train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_datasets, batch_size=64, shuffle=False)
    print(f"{len(train_datasets):,} training windows, {len(val_datasets):,} validation windows")
    return train_loader, val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Let's see what the first pair of input/output sequences look like.
    """)
    return


@app.cell
def _(train_loader):
    next(iter(train_loader))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now define the causal attention mask.  Recall that this mask simply zeroes out the attention weights for future tokens in the sequence. This is done to ensure that the model does not have access to future tokens when making predictions.
    """)
    return


@app.cell
def _(torch):
    torch.arange(10)
    return


@app.cell
def _(device, torch):
    def causal_attention_mask(n_dest, n_src, device):
        i = torch.arange(n_dest, device=device).unsqueeze(1)
        j = torch.arange(n_src, device=device).unsqueeze(0)
        return i >= j


    # Example usage:
    mask = causal_attention_mask(10, 10, device)
    print(mask[2])
    return (causal_attention_mask,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Recall that we also need to define a position embedding.  Here we will use a simple positional encoding corresponding to the embedding of the index of the token in the sequence.
    """)
    return


@app.cell
def _(nn, torch):
    class TokenAndPositionEmbedding(nn.Module):
        def __init__(self, max_len, vocab_size, embed_dim):
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, embed_dim)
            self.pos_emb = nn.Embedding(max_len, embed_dim)

        def forward(self, x):
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            pos_embeddings = self.pos_emb(positions)
            token_embeddings = self.token_emb(x)
            return token_embeddings + pos_embeddings

    return (TokenAndPositionEmbedding,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we define the Transformer block, consisting of, in addition to the usual fully connected layers, also multi-head attention and layer normalization layers.
    """)
    return


@app.cell
def _(causal_attention_mask, nn):
    class TransformerBlock(nn.Module):
        def __init__(self, num_heads, key_dim, embed_dim, ff_dim, dropout_rate=0.1):
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
            self.ln_1 = nn.LayerNorm(embed_dim)
            self.dropout_1 = nn.Dropout(dropout_rate)

            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, embed_dim),
                nn.Dropout(dropout_rate)
            )
            self.ln_2 = nn.LayerNorm(embed_dim)

        def forward(self, x):
            batch_size, seq_len, _ = x.size()
            causal_mask = causal_attention_mask(seq_len, seq_len, x.device)
            attn_output, attn_weights = self.attn(x, x, x, attn_mask=~causal_mask.bool())
            x = self.ln_1(x + self.dropout_1(attn_output))
            ffn_output = self.ffn(x)
            x = self.ln_2(x + ffn_output)
            return x, attn_weights

    return (TransformerBlock,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, let's put it all together into a GPT (Generative Pre-trained Transformer) architecture and train the model using the dataloader defined earlier.
    """)
    return


@app.cell
def _(TokenAndPositionEmbedding, TransformerBlock, nn):
    # GPT-style transformer wrapper
    class GPT(nn.Module):
        def __init__(self, max_len, vocab_size, embed_dim, num_heads, key_dim, ff_dim):
            super().__init__()
            self.embed = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
            self.transformer = TransformerBlock(num_heads, key_dim, embed_dim, ff_dim)
            self.lm_head = nn.Linear(embed_dim, vocab_size)

        def forward(self, x):
            x = self.embed(x)
            x, attn_weights = self.transformer(x)
            logits = self.lm_head(x)
            return logits, attn_weights

    return (GPT,)


@app.cell
def _(torch):
    from tqdm import tqdm

    def evaluate_gpt(model, dataloader, criterion, device):
        """Average next-token cross-entropy (in nats) over a dataloader."""
        model.eval()
        total_loss, n_examples = 0.0, 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                total_loss += loss.item() * inputs.size(0)
                n_examples += inputs.size(0)
        model.train()
        return total_loss / n_examples

    def train_gpt(model, dataloader, val_loader, optimizer, criterion, epochs, device):
        model.to(device)
        # Validation loss of the untrained model (epoch 0), then once per epoch
        history = {"train_loss": [], "val_loss": [evaluate_gpt(model, val_loader, criterion, device)]}
        model.train()

        for epoch in range(epochs):
            total_loss = 0

            data_loader_with_progress = tqdm(
                iterable=dataloader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}"
            )
            for batch_number, (inputs, targets) in enumerate(data_loader_with_progress):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                logits, _ = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                if (batch_number % 100 == 0) or (batch_number == len(dataloader) - 1):
                    data_loader_with_progress.set_postfix(
                        {
                            "avg loss": f"{total_loss/(batch_number+1):.4f}",
                        }
                    )

            history["train_loss"].append(total_loss / len(dataloader))
            history["val_loss"].append(evaluate_gpt(model, val_loader, criterion, device))
            print(f"Epoch {epoch+1}: train loss {history['train_loss'][-1]:.3f}, "
                  f"val loss {history['val_loss'][-1]:.3f}, "
                  f"val perplexity {torch.exp(torch.tensor(history['val_loss'][-1])).item():.1f}")
        return history

    return (train_gpt,)


@app.cell
def _():
    # Hyperparameters
    # Vocabulary and sequence
    VOCAB_SIZE = 10000        # Size of your tokenizer vocab
    MAX_LEN = 128            # Sequence length

    # Embedding and model size
    EMBEDDING_DIM = 256      # Dimension of token/position embeddings
    N_HEADS = 4              # Number of attention heads
    KEY_DIM = 64             # Dimensionality per head (must divide EMBEDDING_DIM)
    FEED_FORWARD_DIM = 1024  # Dimension of feed-forward layer

    # Training
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 3e-3
    return (
        EMBEDDING_DIM,
        EPOCHS,
        FEED_FORWARD_DIM,
        KEY_DIM,
        LEARNING_RATE,
        MAX_LEN,
        N_HEADS,
        VOCAB_SIZE,
    )


@app.cell
def _(
    EMBEDDING_DIM,
    EPOCHS,
    FEED_FORWARD_DIM,
    GPT,
    KEY_DIM,
    LEARNING_RATE,
    MAX_LEN,
    N_HEADS,
    VOCAB_SIZE,
    device,
    nn,
    torch,
    train_gpt,
    train_loader,
    val_loader,
):
    model = GPT(MAX_LEN, VOCAB_SIZE, EMBEDDING_DIM, N_HEADS, KEY_DIM, FEED_FORWARD_DIM)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = train_gpt(model, train_loader, val_loader, optimizer, criterion, epochs=EPOCHS, device=device)
    return history, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Perplexity

    The cross-entropy loss is the average negative log-probability the model assigns to the correct next word. **Perplexity** is its exponential:

    $$\mathrm{PPL} = \exp\Big(-\frac{1}{N}\sum_{t=1}^{N}\log p(x_t \mid x_{<t})\Big)$$

    A perplexity of $k$ means the model is, on average, as uncertain as if it were choosing uniformly among $k$ words. A model that ignores the context and guesses uniformly has perplexity equal to the vocabulary size (10,000 here, dashed line).

    Things to look for in the plot:

    - **Epoch 0** is the untrained model: its perplexity is close to (or above) the vocabulary size.
    - The **validation** curve is the honest one: it is measured on text the model never trained on.
    - When training perplexity keeps falling but validation perplexity stalls or rises, the model is **overfitting**.
    - The training curve is averaged during the epoch with dropout on, so it is only an approximation.
    """)
    return


@app.cell
def _(VOCAB_SIZE, history, np, plt):
    _train_ppl = np.exp(history["train_loss"])
    _val_ppl = np.exp(history["val_loss"])

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.plot(range(1, len(_train_ppl) + 1), _train_ppl, marker="o", label="train")
    _ax.plot(range(0, len(_val_ppl)), _val_ppl, marker="o", label="validation")
    _ax.axhline(VOCAB_SIZE, linestyle="--", color="gray", label=f"uniform guess (vocabulary size, {VOCAB_SIZE:,})")
    _ax.set_yscale("log")
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("perplexity (log scale)")
    _ax.set_title("Perplexity = exp(cross-entropy loss)")
    _ax.grid(alpha=0.3, which="both")
    _ax.legend()
    print(f"Validation perplexity: {_val_ppl[0]:,.0f} before training, {_val_ppl[-1]:,.1f} after training "
          f"(best {_val_ppl.min():,.1f} at epoch {int(_val_ppl.argmin())})")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now use the trained GPT to generate text.  The model will generate a sequence of tokens based on the input prompt. We can use the inverse mapping from our vocabulary to "translate" the tokens to natural text.
    """)
    return


@app.cell
def _(device, torch):
    class TextGenerator:
        def __init__(self, model, index_to_word, top_k=10):
            self.model = model
            self.model.to(device)
            self.index_to_word = index_to_word
            self.word_to_index = {word: idx for idx, word in index_to_word.items()}

        def sample_from(self, probs, temperature):
            probs[1] = 0  # Mask out UNK token (index 1) to prevent generating <UNK>
            probs = torch.nn.functional.softmax(probs/temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            return next_id, probs

        def generate(self, start_prompt, max_tokens, temperature):
            self.model.eval()
            start_tokens = [self.word_to_index.get(w, 1) for w in start_prompt.split()]
            generated_tokens = start_tokens[:]
            info = []

            with torch.no_grad():
                while len(generated_tokens) < max_tokens:
                    x = torch.tensor([generated_tokens], dtype=torch.long)
                    # Keep only the most recent tokens: the position embedding has max_len slots
                    x = x[:, -self.model.embed.pos_emb.num_embeddings:].to(device)
                    logits, attn_weights = self.model(x)
                    last_logits = logits[0, -1] # .cpu().numpy()
                    sample_token, probs = self.sample_from(last_logits, temperature)
                    generated_tokens.append(sample_token)
                    info.append({
                        "prompt": start_prompt,
                        "word_probs": probs,
                        "atts": attn_weights[0].cpu().numpy()
                    })
                    if sample_token == 0:
                        break
            print("GEN", generated_tokens)
            generated_words = [self.index_to_word.get(idx, "<UNK>") for idx in generated_tokens]
            print("generated text:" + " ".join(generated_words))
            return info

    return (TextGenerator,)


@app.cell
def _(TextGenerator, inv_vocab, model):
    text_generator = TextGenerator(model, inv_vocab)
    info = text_generator.generate("captain ", max_tokens=180, temperature=3.0)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
