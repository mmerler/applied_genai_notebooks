import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Module 5: Practical - RNN
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Let's improve the simple bigram text generation model we constructed in Module 1. First we construct the vocabulary from 10000 most frequently occuring words, assigning positions based on the frequency of occurance.
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
    return DataLoader, Dataset, device, nn, re, torch


@app.cell
def _(re):
    # Load and preprocess Count of Monte Cristo
    url = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"
    #url = "https://www.fulltextarchive.com/book/the-count-of-monte-cristo/index.php"

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
    print(encoded)
    return encoded, inv_vocab, vocab


@app.cell
def _(vocab):
    vocab
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Since we are training the model to predict the next word in a sequence, we will construct our training set features based on 30 word sequences from the text. The corresponding labels are the sequences shifted by one word.
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

    train_datasets = TextDataset(encoded)
    train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True)
    return (train_loader,)


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
    We now define the LSTM architecture. Since the LSTM layer is already implemented in PyTorch, we can use it directly. The LSTM layer takes the input sequence and returns the output sequence along with the hidden state. The output is then passed through a fully connected layer to get the final predictions.
    """)
    return


@app.cell
def _(device, nn, torch):
    class LSTMModel(nn.Module):
        def __init__(self, vocab_size=10000, embedding_dim=100, hidden_dim=128):
            super(LSTMModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x, hidden=None):
            x = self.embedding(x)
            x, hidden = self.lstm(x, hidden)
            x = self.fc(x)
            return x, hidden

    model = LSTMModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    return criterion, model, optimizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we train the model for 15 epochs. The loss should decrease over time, indicating that the model is learning to predict the next word in the sequence. After training, we can generate text by providing a seed phrase and letting the model predict the next words based on the learned patterns.
    """)
    return


@app.cell
def _(criterion, device, model, optimizer, train_loader):
    from tqdm import tqdm

    EPOCHS = 15

    # Training loop
    for epoch in range(EPOCHS):
        total_loss = 0
        train_loader_with_progress = tqdm(
            iterable=train_loader, ncols=120, desc=f"Epoch {epoch+1}/{EPOCHS}"
        )
        for batch_number, (inputs, targets) in enumerate(train_loader_with_progress):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_number % 100 == 0) or (batch_number == len(train_loader) - 1):
                train_loader_with_progress.set_postfix(
                    {
                        "avg loss": f"{total_loss/(batch_number+1):.4f}",
                    }
                )
    return


@app.cell
def _(device, inv_vocab, torch, vocab):
    def generate_text(model, seed_text, length=50, temperature=1.0):
        model.eval()
        words = seed_text.lower().split()
        input_ids = [vocab.get(w, vocab["<UNK>"]) for w in words]
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
        hidden = None

        with torch.no_grad():
            for _ in range(length):
                output, hidden = model(input_tensor, hidden)
                logits = output[0, -1] / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
                words.append(inv_vocab.get(next_id, "<UNK>"))

                # Extend input sequence with new token
                input_ids.append(next_id)
                input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

        return " ".join(words)

    return (generate_text,)


@app.cell
def _(generate_text, model, vocab):
    seed = "the sailboat"
    vocab.get(seed)
    print("\nGenerated Text:\n")
    print(generate_text(model, seed, length=150, temperature=1.0))
    return


if __name__ == "__main__":
    app.run()
