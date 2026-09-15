from pathlib import Path

from tokenizer import RegexTokenizer


TRAINING_TEXT = """
Hello! My name is Hari. This is a small training corpus for a byte-level BPE
tokenizer. The tokenizer learns common pieces of text by repeatedly finding the
most frequent adjacent byte pair and merging it into a new token.

Machine learning systems process numbers, not raw human text. A tokenizer
converts text into token IDs. A language model can then operate on those IDs.

The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
Hello world! Hello world! Hello world!
Tokenization is useful for English, numbers, punctuation, and Unicode text.
"""


def main() -> None:
    tokenizer = RegexTokenizer()

    # 320 = 256 initial byte tokens + 64 learned merge tokens.
    tokenizer.train(
        TRAINING_TEXT,
        vocab_size=320,
        special_tokens=["<|endoftext|>"],
    )

    tokenizer.save("tokenizer")

    print("Tokenizer trained successfully.")
    print(f"Learned merges: {len(tokenizer.merges)}")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print("Saved tokenizer.model and tokenizer.vocab.json")


if __name__ == "__main__":
    main()
