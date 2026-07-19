import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os

from utils.tokenizer import CharacterTokenizer
from utils.dataset import ShakespeareDataset
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:",device)
tokenizer = CharacterTokenizer("data/shakespeare.txt")
dataset = ShakespeareDataset(tokenizer)

x,y = dataset.get_batch(
    split = "train",
    batch_size=6,
    context_length=10,
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
m
from models.rmsnorm import RMSNorm

print("\n--- RMSNorm Test ---")

d_model = 256

norm = RMSNorm(d_model).to(device)

test_input = torch.randn(
    4,
    8,
    d_model,
    device=device
)

output = norm(test_input)

print("Input shape :", test_input.shape)
print("Output shape:", output.shape)

print("Input RMS :", test_input.pow(2).mean(dim=-1).sqrt().mean().item())
print("Output RMS:", output.pow(2).mean(dim=-1).sqrt().mean().item())