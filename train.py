import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os



# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")
# if device == "cuda":
#     print(f"GPU: {torch.cuda.get_device_name(0)}")

from utils.tokenizer import CharacterTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:",device)
tokenizer = CharacterTokenizer("data/shakespeare.txt")
print()
print("Vocabulary size:",tokenizer.vocab_size)
print()
print("First 20 characters:")
print(tokenizer.chars[:20])
print()
sample="Hello"
encoded = tokenizer.encode(sample)
print("Sample:",sample)
print("Encoded:",encoded)
print("Decoded:",tokenizer.decode(encoded))