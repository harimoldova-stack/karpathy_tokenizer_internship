import torch
from model import TinyGPT


vocab_size = 30

model = TinyGPT(
    vocab_size=vocab_size,
    embedding_size=64,
    block_size=128
)

token_ids = torch.randint(
    0,
    vocab_size,
    (2, 10)
)

output = model(token_ids)

print("Input shape:", token_ids.shape)
print("Output shape:", output.shape)

expected_shape = (2, 10, vocab_size)

if output.shape == expected_shape:
    print("Model test PASSED!")
else:
    print("Model test FAILED!")