from tokenizer import BasicTokenizer, RegexTokenizer, get_stats, merge


def test_get_stats() -> None:
    assert get_stats([1, 2, 1, 2]) == {(1, 2): 2, (2, 1): 1}


def test_merge() -> None:
    assert merge([1, 2, 1, 2], (1, 2), 99) == [99, 99]


def test_basic_round_trip() -> None:
    text = "Hello hello! 123 🍕"
    tokenizer = BasicTokenizer()
    tokenizer.train(text * 10, vocab_size=280)

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded == text


def test_regex_round_trip() -> None:
    text = "Hello world! Привет नमस्ते 123 🍕"
    tokenizer = RegexTokenizer()
    tokenizer.train(
        text * 20,
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
    )

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded == text


def test_special_token() -> None:
    tokenizer = RegexTokenizer()
    tokenizer.train(
        "hello world " * 30,
        vocab_size=280,
        special_tokens=["<|endoftext|>"],
    )

    text = "hello <|endoftext|> world"
    ids = tokenizer.encode(text)

    special_id = tokenizer.special_tokens["<|endoftext|>"]
    assert special_id in ids
    assert tokenizer.decode(ids) == text


if __name__ == "__main__":
    test_get_stats()
    test_merge()
    test_basic_round_trip()
    test_regex_round_trip()
    test_special_token()
    print("ALL TESTS PASSED")
