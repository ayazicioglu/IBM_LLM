import torchtext
print(torchtext.__version__)
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper, Mapper
import torchtext

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random

sentences = [
    "If you want to know what a man's like, take a good look at how he treats his inferiors, not his equals.",
    "Fame's a fickle friend, Harry.",
    "It is our choices, Harry, that show what we truly are, far more than our abilities.",
    "Soon we must all face the choice between what is right and what is easy.",
    "Youth can not know how age thinks and feels. But old men are guilty if they forget what it was to be young.",
    "You are awesome!"
]

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

# Create an instance of your custom dataset
custom_dataset = CustomDataset(sentences)

# Define batch size
batch_size = 2

# Create a DataLoader
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Iterate through the DataLoader
for batch in dataloader:
    print(batch)

sentences = [
    "If you want to know what a man's like, take a good look at how he treats his inferiors, not his equals.",
    "Fame's a fickle friend, Harry.",
    "It is our choices, Harry, that show what we truly are, far more than our abilities.",
    "Soon we must all face the choice between what is right and what is easy.",
    "Youth can not know how age thinks and feels. But old men are guilty if they forget what it was to be young.",
    "You are awesome!"
]

# Define a custom data set
class CustomDataset(Dataset):
    def __init__(self, sentences, tokenizer, vocab):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.sentences[idx])
        # Convert tokens to tensor indices using vocab
        tensor_indices = [self.vocab[token] for token in tokens]
        return torch.tensor(tensor_indices)

# Tokenizer
tokenizer = get_tokenizer("basic_english")



# Build vocabulary
vocab = build_vocab_from_iterator(map(tokenizer, sentences))

# Create an instance of your custom data set with the tokenizer and vocab
custom_dataset = CustomDataset(sentences, tokenizer, vocab)

print("Custom Dataset Length:", len(custom_dataset))
print("Sample Items:")
for i in range(6):
    sample_item = custom_dataset[i]
    print(f"Item {i + 1}: {sample_item}")



    