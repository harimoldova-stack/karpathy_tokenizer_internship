import torch
import torch.nn.functional as F
from pathlib import Path

from tokenizer import CharacterTokenizer
from model import TinyGPT


# -----------------------------
# Settings
# -----------------------------

batch_size = 4
block_size = 64
learning_rate = 0.001
steps = 5000
embedding_size = 64
random_seed = 42


# -----------------------------
# Reproducibility
# -----------------------------

torch.manual_seed(random_seed)


# -----------------------------
# Load data
# -----------------------------

base_path = Path(__file__).parent

train_path = base_path / "data" / "train.txt"
val_path = base_path / "data" / "val.txt"

with open(train_path, "r", encoding="utf-8") as file:
    train_text = file.read()

with open(val_path, "r", encoding="utf-8") as file:
    val_text = file.read()


# -----------------------------
# Create tokenizer
# -----------------------------

# Build vocabulary from both datasets
all_text = train_text + val_text

tokenizer = CharacterTokenizer(all_text)

vocab_size = tokenizer.vocab_size

print("Vocabulary size:", vocab_size)
print("Random seed:", random_seed)


# -----------------------------
# Encode datasets
# -----------------------------

train_data = torch.tensor(
    tokenizer.encode(train_text),
    dtype=torch.long
)

val_data = torch.tensor(
    tokenizer.encode(val_text),
    dtype=torch.long
)


# -----------------------------
# Create batches
# -----------------------------

def get_batch(data):

    starts = torch.randint(
        0,
        len(data) - block_size - 1,
        (batch_size,)
    )

    inputs = torch.stack([
        data[i:i + block_size]
        for i in starts
    ])

    targets = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in starts
    ])

    return inputs, targets


# -----------------------------
# Create model
# -----------------------------

model = TinyGPT(
    vocab_size=vocab_size,
    embedding_size=embedding_size,
    block_size=block_size
)


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# -----------------------------
# Calculate validation loss
# -----------------------------

def calculate_loss(data):

    inputs, targets = get_batch(data)

    with torch.no_grad():

        logits = model(inputs)

        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )

    return loss.item()


# -----------------------------
# Training
# -----------------------------

print("Starting training...")

for step in range(steps):

    inputs, targets = get_batch(train_data)

    logits = model(inputs)

    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1)
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 50 == 0:

        train_loss = loss.item()
        val_loss = calculate_loss(val_data)

        print(
            f"Step {step}: "
            f"Train Loss = {train_loss:.4f}, "
            f"Validation Loss = {val_loss:.4f}"
        )


# -----------------------------
# Save checkpoint
# -----------------------------

checkpoint_path = base_path / "tiny_code_gpt_checkpoint.pt"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "embedding_size": embedding_size,
        "block_size": block_size,
        "random_seed": random_seed,
    },
    checkpoint_path
)

print("Training complete!")
print("Checkpoint saved:", checkpoint_path)