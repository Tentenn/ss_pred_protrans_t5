"""
This code trains the CNN for 3-State secondary structure prediction
Using ProtTrans T5 per protein embeddings.
"""
import torch
from torch import nn
import torch.optim as optim
from typing import Any
from tqdm.auto import tqdm
import json
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import T5Encoder, T5Tokenizer
from pathlib import Path
from pyfaidx import Fasta
from typing import Dict, Tuple, List
import numpy as np
import re
import h5py


class EmbedDataset(Dataset):
    def __init__(self, embed_path: str, labels_dict: Dict[str, int],
                 device: torch.device):
        self.device = device
        self.embeddings = self.load_embeds(embed_path)
        self.labels_dict = labels_dict
        self.headers = tuple(self.embeddings.keys())
        assert len(self.headers) == len(self.labels_dict.keys())

    def load_embeds(self, embed_path: str):
        with h5py.File(embed_path, 'r') as f:
            d = {seq_identifier: np.array(f[seq_identifier]) for seq_identifier in f.keys()}
        return d

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        ## given an index, return embedding and label sequence
        header = self.headers[index]
        embedding = torch.tensor(self.embeddings[header]).to(self.device)
        label = self.labels_dict[header]
        return embedding, label

    def __len__(self) -> int:
        return len(self.headers)


def get_dataloader(embed_path: str, labels_dict: Dict[str, int],
                   batch_size: int, device: torch.device,
                   seed: int) -> DataLoader:
    torch.manual_seed(seed)
    dataset = EmbedDataset(embed_path=embed_path,
                           labels_dict=labels_dict,
                           device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def main_training_loop():
    pass


def train():
    pass


def validate():
    pass

if __name__ == "__main__":
    EPOCHS = 10
    BATCHSIZE = 20

    train_embeds_path = ""
    val_embeds_path = ""

    ## Load model
    ## Load Dataset (train and validate)
    ##

