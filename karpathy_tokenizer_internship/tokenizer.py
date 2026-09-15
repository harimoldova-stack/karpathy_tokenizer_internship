from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import regex


Pair = Tuple[int, int]


def get_stats(ids: List[int]) -> Dict[Pair, int]:
    """Count adjacent token pairs."""
    counts: Dict[Pair, int] = {}
    for a, b in zip(ids, ids[1:]):
        pair = (a, b)
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: List[int], pair: Pair, new_id: int) -> List[int]:
    """Replace every non-overlapping occurrence of pair with new_id."""
    out: List[int] = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BasicTokenizer:
    """Simple byte-level BPE tokenizer.

    This class demonstrates the core BPE algorithm without regex
    pre-tokenization or special-token handling.
    """

    def __init__(self) -> None:
        self.merges: Dict[Pair, int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        ids = list(text.encode("utf-8"))
        num_merges = vocab_size - 256

        for _ in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            new_id = 256 + len(self.merges)

            ids = merge(ids, best_pair, new_id)
            self.merges[best_pair] = new_id

            self.vocab[new_id] = (
                self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            )

    def encode(self, text: str) -> List[int]:
        ids = list(text.encode("utf-8"))

        if not self.merges:
            return ids

        # Apply merges in training order.
        for pair, new_id in self.merges.items():
            ids = merge(ids, pair, new_id)

        return ids

    def decode(self, ids: Iterable[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")


class RegexTokenizer(BasicTokenizer):
    """Byte-level BPE with regex pre-tokenization and special tokens."""

    # A GPT-2-style educational pattern. The third-party `regex` package
    # supports Unicode properties such as \p{L} and \p{N}.
    PATTERN = (
        r"""'(?:[sdmt]|ll|ve|re)"""
        r"""| ?\p{L}+"""
        r"""| ?\p{N}+"""
        r"""| ?[^\s\p{L}\p{N}]+"""
        r"""|\s+(?!\S)"""
        r"""|\s+"""
    )

    def __init__(self) -> None:
        super().__init__()
        self.pattern = self.PATTERN
        self.compiled_pattern = regex.compile(self.pattern)
        self.special_tokens: Dict[str, int] = {}

    def register_special_tokens(self, special_tokens: List[str]) -> None:
        start = max(self.vocab.keys()) + 1
        for token in special_tokens:
            if token not in self.special_tokens:
                self.special_tokens[token] = start
                self.vocab[start] = token.encode("utf-8")
                start += 1

    def train(
        self,
        text: str,
        vocab_size: int,
        special_tokens: List[str] | None = None,
    ) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        # Split text into pieces and train BPE over all pieces together.
        chunks = self.compiled_pattern.findall(text)
        ids = [list(chunk.encode("utf-8")) for chunk in chunks]

        num_merges = vocab_size - 256
        for _ in range(num_merges):
            stats: Dict[Pair, int] = {}
            for chunk_ids in ids:
                local = get_stats(chunk_ids)
                for pair, count in local.items():
                    stats[pair] = stats.get(pair, 0) + count

            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            new_id = 256 + len(self.merges)

            ids = [merge(chunk_ids, best_pair, new_id) for chunk_ids in ids]
            self.merges[best_pair] = new_id
            self.vocab[new_id] = (
                self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            )

        # Special tokens are assigned after learned BPE tokens so their IDs
        # cannot collide with merge IDs.
        if special_tokens:
            self.register_special_tokens(special_tokens)

    def encode_ordinary(self, text: str) -> List[int]:
        ids: List[int] = []
        for chunk in self.compiled_pattern.findall(text):
            chunk_ids = list(chunk.encode("utf-8"))

            # Only apply merges relevant to this chunk.
            while len(chunk_ids) >= 2:
                stats = get_stats(chunk_ids)
                applicable = [
                    (pair, new_id)
                    for pair, new_id in self.merges.items()
                    if pair in stats
                ]
                if not applicable:
                    break

                # Earlier learned merges have priority.
                pair, new_id = min(
                    applicable, key=lambda item: list(self.merges).index(item[0])
                )
                chunk_ids = merge(chunk_ids, pair, new_id)

            ids.extend(chunk_ids)
        return ids

    def encode(self, text: str) -> List[int]:
        if not self.special_tokens:
            return self.encode_ordinary(text)

        # Longest special token first prevents a shorter token from
        # accidentally matching inside a longer special token.
        specials = sorted(self.special_tokens, key=len, reverse=True)
        parts = regex.split("(" + "|".join(regex.escape(s) for s in specials) + ")", text)

        ids: List[int] = []
        for part in parts:
            if part == "":
                continue
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids

    def save(self, prefix: str) -> None:
        model_path = Path(prefix + ".model")
        vocab_path = Path(prefix + ".vocab.json")

        model = {
            "version": 1,
            "pattern": self.pattern,
            "merges": [[list(pair), new_id] for pair, new_id in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")

        readable_vocab = {
            str(i): self.vocab[i].decode("utf-8", errors="replace")
            for i in sorted(self.vocab)
        }
        vocab_path.write_text(
            json.dumps(readable_vocab, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self, prefix: str) -> None:
        model_path = Path(prefix + ".model")
        data = json.loads(model_path.read_text(encoding="utf-8"))

        self.pattern = data["pattern"]
        self.compiled_pattern = regex.compile(self.pattern)
        self.merges = {
            (int(pair[0]), int(pair[1])): int(new_id)
            for pair, new_id in data["merges"]
        }
        self.special_tokens = {
            str(token): int(token_id)
            for token, token_id in data.get("special_tokens", {}).items()
        }

        self.vocab = {i: bytes([i]) for i in range(256)}
        for pair, new_id in self.merges.items():
            self.vocab[new_id] = (
                self.vocab[pair[0]] + self.vocab[pair[1]]
            )

        for token, token_id in self.special_tokens.items():
            self.vocab[token_id] = token.encode("utf-8")
