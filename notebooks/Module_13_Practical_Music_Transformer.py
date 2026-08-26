import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Module 8: Practical - Music Generation Transformer""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    In this practical, we'll build a transformer model for music generation. Similar to text generation,
    we'll train a model to predict the next note in a sequence, enabling it to compose original melodies.

    We'll use a simple piano roll representation where each note is represented by its MIDI pitch value (0-127).
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    torch.manual_seed(42)
    np.random.seed(42)

    # Check which GPU is available
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    print(f"Using device: {device}")
    return DataLoader, Dataset, device, nn, plt, torch


@app.cell
def _():
    import glob
    import os
    import pretty_midi

    def load_midi_files(midi_dir=None):
        """Load MIDI files from a directory and extract note sequences"""
        all_notes = []

        if os.path.exists(midi_dir):
            midi_files = glob.glob(os.path.join(midi_dir, "*.mid")) + \
                         glob.glob(os.path.join(midi_dir, "*.midi"))

            print(f"Found {len(midi_files)} MIDI files")

            for midi_file in midi_files[:20]:  # Limit to first 20 files
                try:
                    midi_data = pretty_midi.PrettyMIDI(midi_file)

                    # Extract notes from all instruments
                    for instrument in midi_data.instruments:
                        if not instrument.is_drum:  # Skip drum tracks
                            # Sort notes by start time
                            notes = sorted(instrument.notes, key=lambda x: x.start)

                            # Extract pitch values
                            for note in notes:
                                all_notes.append(note.pitch)

                except Exception as e:
                    print(f"Error loading {midi_file}: {e}")
                    continue

            if len(all_notes) > 0:
                print(f"Extracted {len(all_notes)} notes from MIDI files")
                return all_notes

        print("Loading MIDI failed")

    midi_directory = "../data/maestro-v1.0.0/2015"  # Set to "./midi_files" or your MIDI folder path

    music_data = load_midi_files(midi_directory)
    print(f"Total musical events: {len(music_data)}")
    print(f"Sample notes: {music_data[:30]}")
    print(f"Note range: {min(music_data)} to {max(music_data)}")
    return (music_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Vocabulary and Tokenization

    For music, our vocabulary consists of:
    - MIDI note values (0-127)
    - Special tokens: <PAD> (padding), <START> (sequence start), <END> (sequence end)
    """
    )
    return


@app.cell
def _(music_data):
    # Create vocabulary for music
    # MIDI notes range from 0-127, we'll add special tokens
    VOCAB_SIZE = 131  # 128 MIDI notes + <PAD>=128, <START>=129, <END>=130
    PAD_TOKEN = 128
    START_TOKEN = 129
    END_TOKEN = 130

    # Map for decoding
    def decode_note(token):
        if token == PAD_TOKEN:
            return "<PAD>"
        elif token == START_TOKEN:
            return "<START>"
        elif token == END_TOKEN:
            return "<END>"
        else:
            return f"Note{token}"

    # Encode the music data (already in MIDI format)
    encoded_music = music_data.copy()

    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Encoded sequence length: {len(encoded_music)}")
    return VOCAB_SIZE, decode_note, encoded_music


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Sequence Creation

    We'll create fixed-length sequences of musical events. Each sequence will be used to predict
    the next note, similar to language modeling.
    """
    )
    return


@app.cell
def _(DataLoader, Dataset, encoded_music, torch):
    # Create sequences for music generation
    SEQ_LEN = 32  # Sequence length for music patterns

    class MusicDataset(Dataset):
        def __init__(self, data, seq_len):
            self.data = data
            self.seq_len = seq_len

        def __len__(self):
            return len(self.data) - self.seq_len

        def __getitem__(self, idx):
            return (
                torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long),
                torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
            )

    music_dataset = MusicDataset(encoded_music, SEQ_LEN)
    music_loader = DataLoader(music_dataset, batch_size=32, shuffle=True)

    print(f"Dataset size: {len(music_dataset)} sequences")
    print(f"Batch size: 32")
    return (music_loader,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Let's examine a sample batch of sequences:""")
    return


@app.cell
def _(music_loader):
    sample_batch = next(iter(music_loader))
    print(f"Input shape: {sample_batch[0].shape}")
    print(f"Target shape: {sample_batch[1].shape}")
    print(f"\nFirst sequence (input): {sample_batch[0][0][:10].tolist()}")
    print(f"First sequence (target): {sample_batch[1][0][:10].tolist()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Transformer Architecture Components

    We'll use the same transformer architecture as for text generation:
    1. Causal attention mask (prevent looking ahead)
    2. Token and position embeddings
    3. Multi-head attention
    4. Feed-forward layers
    """
    )
    return


@app.cell
def _(device, torch):
    def causal_attention_mask(n_dest, n_src, device):
        """Create a causal mask to prevent attending to future positions"""
        i = torch.arange(n_dest, device=device).unsqueeze(1)
        j = torch.arange(n_src, device=device).unsqueeze(0)
        return i >= j

    # Example usage:
    mask = causal_attention_mask(10, 10, device)
    print("Causal mask (position 2):", mask[2].tolist())
    return (causal_attention_mask,)


@app.cell
def _(nn, torch):
    class TokenAndPositionEmbedding(nn.Module):
        """Embed both the note tokens and their positions in the sequence"""
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


@app.cell
def _(causal_attention_mask, nn):
    class TransformerBlock(nn.Module):
        """Single transformer block with multi-head attention and feed-forward layers"""
        def __init__(self, num_heads, embed_dim, ff_dim, dropout_rate=0.1):
            super().__init__()
            self.attn = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout_rate, batch_first=True
            )
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

            # Multi-head attention with causal masking
            attn_output, attn_weights = self.attn(x, x, x, attn_mask=~causal_mask.bool())
            x = self.ln_1(x + self.dropout_1(attn_output))

            # Feed-forward network
            ffn_output = self.ffn(x)
            x = self.ln_2(x + ffn_output)

            return x, attn_weights
    return (TransformerBlock,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Music Transformer Model

    We now assemble all components into a complete music generation transformer.
    This model will learn to predict the next note given a sequence of previous notes.
    """
    )
    return


@app.cell
def _(TokenAndPositionEmbedding, TransformerBlock, nn):
    class MusicTransformer(nn.Module):
        """Complete transformer model for music generation"""
        def __init__(self, max_len, vocab_size, embed_dim, num_heads, ff_dim, num_layers=2):
            super().__init__()
            self.embed = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)

            # Stack multiple transformer blocks
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(num_heads, embed_dim, ff_dim)
                for _ in range(num_layers)
            ])

            self.lm_head = nn.Linear(embed_dim, vocab_size)

        def forward(self, x):
            x = self.embed(x)

            attn_weights_list = []
            for transformer in self.transformer_blocks:
                x, attn_weights = transformer(x)
                attn_weights_list.append(attn_weights)

            logits = self.lm_head(x)
            return logits, attn_weights_list
    return (MusicTransformer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Training Function

    We'll train the model using cross-entropy loss to predict the next note in each sequence.
    """
    )
    return


@app.cell
def _():
    from tqdm import tqdm

    def train_music_transformer(model, dataloader, optimizer, criterion, epochs, device):
        """Train the music transformer model"""
        model.to(device)
        model.train()

        loss_history = []

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

                # Compute loss
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if (batch_number % 50 == 0) or (batch_number == len(dataloader) - 1):
                    avg_loss = total_loss / (batch_number + 1)
                    data_loader_with_progress.set_postfix({"avg loss": f"{avg_loss:.4f}"})

            epoch_loss = total_loss / len(dataloader)
            loss_history.append(epoch_loss)
            print(f"Epoch {epoch+1}/{epochs} - Average Loss: {epoch_loss:.4f}")

        return loss_history
    return (train_music_transformer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Hyperparameters

    Let's define the hyperparameters for our music transformer:
    """
    )
    return


@app.cell
def _(VOCAB_SIZE):
    # Hyperparameters
    MAX_LEN = 128           # Maximum sequence length
    EMBEDDING_DIM = 128     # Embedding dimension
    N_HEADS = 4             # Number of attention heads
    FEED_FORWARD_DIM = 512  # Feed-forward layer dimension
    NUM_LAYERS = 3          # Number of transformer blocks
    DROPOUT_RATE = 0.1

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-3

    print(f"Model Configuration:")
    print(f"  Vocabulary Size: {VOCAB_SIZE}")
    print(f"  Max Sequence Length: {MAX_LEN}")
    print(f"  Embedding Dimension: {EMBEDDING_DIM}")
    print(f"  Number of Heads: {N_HEADS}")
    print(f"  Feed-Forward Dimension: {FEED_FORWARD_DIM}")
    print(f"  Number of Layers: {NUM_LAYERS}")
    return EMBEDDING_DIM, FEED_FORWARD_DIM, MAX_LEN, NUM_LAYERS, N_HEADS


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Model Training

    Now let's instantiate and train the model:
    """
    )
    return


@app.cell
def _(
    EMBEDDING_DIM,
    FEED_FORWARD_DIM,
    MAX_LEN,
    MusicTransformer,
    NUM_LAYERS,
    N_HEADS,
    VOCAB_SIZE,
    device,
    music_loader,
    nn,
    torch,
    train_music_transformer,
):
    # Create model
    music_model = MusicTransformer(
        MAX_LEN, VOCAB_SIZE, EMBEDDING_DIM, N_HEADS, FEED_FORWARD_DIM, NUM_LAYERS
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(music_model.parameters(), lr=1e-3)

    # Train the model
    loss_history = train_music_transformer(
        music_model, music_loader, optimizer, criterion, epochs=15, device=device
    )
    return loss_history, music_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Visualize Training Progress

    Let's plot the training loss over epochs:
    """
    )
    return


@app.cell
def _(loss_history, plt):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Music Generation

    Now we can use the trained model to generate new musical sequences!
    The generator will predict the next note based on the current sequence.
    """
    )
    return


@app.cell
def _(decode_note, device, torch):
    class MusicGenerator:
        """Generate music sequences using the trained transformer"""
        def __init__(self, model, vocab_size, temperature=1.0):
            self.model = model
            self.model.to(device)
            self.vocab_size = vocab_size
            self.temperature = temperature

        def sample_from_logits(self, logits, temperature):
            """Sample next note from logits with temperature scaling"""
            # Apply temperature
            probs = torch.nn.functional.softmax(logits / temperature, dim=-1)

            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1).item()
            return next_token, probs

        def generate(self, seed_sequence, max_length=100, temperature=1.0):
            """
            Generate a music sequence

            Args:
                seed_sequence: Initial sequence of MIDI notes (list of ints)
                max_length: Maximum length of generated sequence
                temperature: Sampling temperature (higher = more random)
            """
            self.model.eval()
            generated = seed_sequence.copy()

            with torch.no_grad():
                for _ in range(max_length):
                    # Prepare input (use last 32 notes as context)
                    context = generated[-32:]
                    x = torch.tensor([context], dtype=torch.long).to(device)

                    # Get prediction
                    logits, _ = self.model(x)
                    last_logits = logits[0, -1]

                    # Sample next note
                    next_note, probs = self.sample_from_logits(last_logits, temperature)

                    # Stop if we generate a pad token
                    if next_note >= 128:
                        break

                    generated.append(next_note)

            # Decode to note names
            generated_notes = [decode_note(n) for n in generated]

            print(f"Generated {len(generated)} notes")
            print(f"MIDI sequence: {generated}")

            return generated, generated_notes
    return (MusicGenerator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Generate Music

    Let's generate some music! We'll start with a seed sequence (C major scale start)
    and let the model continue the melody.
    """
    )
    return


@app.cell
def _(MusicGenerator, VOCAB_SIZE, music_model):
    # Create generator
    generator = MusicGenerator(music_model, VOCAB_SIZE)

    # Seed sequence: Start of C major scale
    seed = [60, 62, 64, 65]  # C, D, E, F

    print("Seed sequence:", seed)
    print("\nGenerating music with low temperature (more predictable)...")
    generated_low, notes_low = generator.generate(seed, max_length=50, temperature=0.5)

    print("\nGenerating music with medium temperature (balanced)...")
    generated_med, notes_med = generator.generate(seed, max_length=50, temperature=1.0)

    print("\nGenerating music with high temperature (more random)...")
    generated_high, notes_high = generator.generate(seed, max_length=50, temperature=1.5)
    return generated_high, generated_low, generated_med


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Visualize Generated Music

    Let's create a piano roll visualization of the generated sequences:
    """
    )
    return


@app.cell
def _(generated_high, generated_low, generated_med, plt):
    def plot_piano_roll(sequence, title="Piano Roll"):
        """Plot a piano roll visualization of a MIDI sequence"""
        fig, ax = plt.subplots(figsize=(15, 4))

        # Filter out special tokens (>127)
        notes = [n for n in sequence if n < 128]

        # Create piano roll
        for i, note in enumerate(notes):
            if note > 0:  # Skip rests
                ax.plot([i, i+1], [note, note], 'b-', linewidth=8)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('MIDI Note')
        ax.set_title(title)
        ax.set_ylim(50, 80)  # Focus on typical piano range
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    # Plot all three generations
    fig1 = plot_piano_roll(generated_low, "Generated Music (Temperature=0.5)")
    plt.show()

    fig2 = plot_piano_roll(generated_med, "Generated Music (Temperature=1.0)")
    plt.show()

    fig3 = plot_piano_roll(generated_high, "Generated Music (Temperature=1.5)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Export to MIDI and Audio

    Now let's export the generated music to MIDI files that can be played!
    """
    )
    return


@app.cell
def _():
    import mido
    from mido import Message, MidiFile, MidiTrack

    def export_to_midi(notes, filename="generated_music.mid", tempo=500000, note_duration=480):
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)

        # Set tempo
        track.append(mido.MetaMessage('set_tempo', tempo=tempo))

        # Add notes
        for note in notes:
            if 0 < note < 128:  # Valid MIDI note
                track.append(Message('note_on', note=int(note), velocity=64, time=0))
                track.append(Message('note_off', note=int(note), velocity=64, time=note_duration))

        mid.save(filename)
        print(f"✓ MIDI file saved as '{filename}'")
        return filename
    return (export_to_midi,)


@app.cell
def _(export_to_midi, generated_high, generated_low, generated_med):
    # Export to file
    export_to_midi(generated_low, "generated_low_temp.mid", tempo=450000)
    export_to_midi(generated_med, "generated_med_temp.mid", tempo=450000)
    export_to_midi(generated_high, "generated_high_temp.mid", tempo=450000)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
