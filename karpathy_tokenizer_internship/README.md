# Byte-Level BPE Tokenizer — Internship Project

A from-scratch educational implementation of a byte-level Byte Pair Encoding (BPE)
tokenizer, inspired by Andrej Karpathy's tokenizer/minBPE tutorials.

## What this project implements

- UTF-8 text -> bytes
- Initial vocabulary of 256 byte tokens
- Counting adjacent token pairs
- Selecting the most frequent pair
- BPE merge operations
- Tokenizer training
- `encode()` text -> token IDs
- `decode()` token IDs -> text
- Regex pre-tokenization
- Special tokens
- Save/load tokenizer files
- Basic tests

## Software

- Python 3.10+ recommended
- VS Code recommended
- Git
- A GitHub account for the repository

## Setup

### Windows

Open PowerShell in this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, you can use:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Quick test

```powershell
python test_tokenizer.py
```

You should see all tests pass.

## Train a tokenizer

```powershell
python train_tokenizer.py
```

This creates:

- `tokenizer.model` — vocabulary/merge information
- `tokenizer.vocab.json` — human-readable vocabulary

## Use the trained tokenizer

```powershell
python demo.py
```

## Suggested internship explanation

The tokenizer starts with 256 byte values. During training it repeatedly finds
the most frequent adjacent pair and merges it into a new token. Encoding converts
text into UTF-8 bytes and applies the learned merges in the learned order.
Decoding reverses the process by mapping token IDs back to their byte strings
and decoding UTF-8.

This is an educational implementation; it is not intended to reproduce the
exact production tokenizer used by a specific GPT model.
