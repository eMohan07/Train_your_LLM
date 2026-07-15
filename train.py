import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import time
# import os

from utils.tokenizer import CharacterTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:",device)
tokenizer = CharacterTokenizer("data/shakespeare.txt")
dataset = ShakespeareDataset(tokenizer)

x,y = dataset.get_batch(
    split = "train",
    batch_size=4,
    context_length=8,
    device=device,
)
print()

print("Input Shape:",x.shape)
print("Target Shape:",y.shape)
print()

print("Input")
print(tokenizer.decode(x[0].tolist()))
print()
print("Target")
print(tokenizer.decode(y[0].tolist()))

# print("Vocabulary size:",tokenizer.vocab_size)
# print()
# print("First 20 characters:")
# print(tokenizer.chars[:20])
# print()
# sample="Hello"
# encoded = tokenizer.encode(sample)
# print("Sample:",sample)
# print("Encoded:",encoded)
# print("Decoded:",tokenizer.decode(encoded))