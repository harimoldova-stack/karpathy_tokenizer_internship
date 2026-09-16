# Tiny Code GPT

A small GPT-style Transformer trained from scratch to generate Python code.

## Project Goal

The goal of this project is to build a small language model that learns patterns from a focused Python code dataset and generates code from a partial function prompt.

This project was built as part of the Tiny Code GPT project.

## Project Structure

```text
tiny_code_gpt/
├── data/
│   ├── code.txt
│   ├── train.txt
│   └── val.txt
├── prepare_data.py
├── tokenizer.py
├── test_tokenizer.py
├── model.py
├── test_model.py
├── train.py
├── generate.py
└── README.md