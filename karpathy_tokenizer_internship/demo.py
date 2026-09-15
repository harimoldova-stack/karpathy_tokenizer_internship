from tokenizer import RegexTokenizer


def main() -> None:
    tokenizer = RegexTokenizer()
    tokenizer.load("tokenizer")

    samples = [
        "Hello world!",
        "The quick brown fox.",
        "Numbers: 12345",
        "Unicode: Привет नमस्ते 你好 🍕",
        "Hello <|endoftext|> Goodbye",
    ]

    for text in samples:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        print("=" * 60)
        print("Original :", text)
        print("Token IDs :", tokens)
        print("Decoded   :", decoded)
        print("Round trip:", text == decoded)


if __name__ == "__main__":
    main()
