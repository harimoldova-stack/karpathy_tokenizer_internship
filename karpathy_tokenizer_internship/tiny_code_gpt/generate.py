import ast
import torch
from pathlib import Path

from tokenizer import CharacterTokenizer
from model import TinyGPT


base_path = Path(__file__).parent

train_path = base_path / "data" / "train.txt"
val_path = base_path / "data" / "val.txt"
checkpoint_path = base_path / "tiny_code_gpt_checkpoint.pt"


# Load dataset
with open(train_path, "r", encoding="utf-8") as file:
    train_text = file.read()

with open(val_path, "r", encoding="utf-8") as file:
    val_text = file.read()

all_text = train_text + val_text


# Create tokenizer
tokenizer = CharacterTokenizer(all_text)


# Load checkpoint
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu"
)


# Rebuild model
model = TinyGPT(
    vocab_size=checkpoint["vocab_size"],
    embedding_size=checkpoint["embedding_size"],
    block_size=checkpoint["block_size"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


def generate(
    prompt,
    max_new_tokens=40,
    temperature=0.2,
    top_k=5
):

    token_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long
    )

    for _ in range(max_new_tokens):

        token_ids = token_ids[:, -model.block_size:]

        with torch.no_grad():
            logits = model(token_ids)

        logits = logits[:, -1, :]

        # Temperature
        logits = logits / temperature

        # Top-k sampling
        if top_k is not None:

            values, indices = torch.topk(
                logits,
                min(top_k, logits.size(-1))
            )

            filtered_logits = torch.full_like(
                logits,
                float("-inf")
            )

            filtered_logits.scatter_(
                1,
                indices,
                values
            )

            logits = filtered_logits

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        token_ids = torch.cat(
            [token_ids, next_token],
            dim=1
        )

        generated_text = tokenizer.decode(
            token_ids[0].tolist()
        )

        continuation = generated_text[len(prompt):]

        if "\ndef " in continuation:
            break

    return tokenizer.decode(
        token_ids[0].tolist()
    )


def check_python_syntax(code):

    try:
        ast.parse(code)
        return True

    except SyntaxError:
        return False


# Test prompts
prompts = [
    "def square(n):",
    "def add(a, b):",
    "def multiply(a, b):"
]


# Run all tests
for prompt in prompts:

    generated_code = generate(
        prompt,
        max_new_tokens=40,
        temperature=0.2,
        top_k=5
    )

    # Stop if another function appears
    if "\ndef " in generated_code:

        generated_code = generated_code.split(
            "\ndef ",
            1
        )[0]

    print("\n" + "=" * 50)

    print("\nPrompt:")
    print(prompt)

    print("\nGenerated code:")
    print(generated_code)

    print("\nSampling settings:")
    print("Temperature:", 0.2)
    print("Top-k:", 5)

    print("\nSyntax check:")

    if check_python_syntax(generated_code):

        print(
            "PASSED - Generated code has valid Python syntax."
        )

    else:

        print(
            "FAILED - Generated code has invalid Python syntax."
        )


print("\n" + "=" * 50)
print("All generation tests completed.")