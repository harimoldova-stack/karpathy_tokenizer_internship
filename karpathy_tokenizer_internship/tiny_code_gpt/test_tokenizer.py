from pathlib import Path
from tokenizer import CharacterTokenizer


# Load our training data
data_path = Path(__file__).parent / "data" / "train.txt"

with open(data_path, "r", encoding="utf-8") as file:
    text = file.read()


# Create tokenizer
tokenizer = CharacterTokenizer(text)


# Test text
sample = "def add(a, b):"


# Encode
encoded = tokenizer.encode(sample)

print("Original:")
print(sample)

print("\nEncoded:")
print(encoded)

# Decode
decoded = tokenizer.decode(encoded)

print("\nDecoded:")
print(decoded)

# Check
if sample == decoded:
    print("\nTokenizer test PASSED!")
else:
    print("\nTokenizer test FAILED!")