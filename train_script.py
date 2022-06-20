"""
This code trains the CNN for 3-State secondary structure prediction
Using ProtTrans T5 per residue embeddings.
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

import utils
from ConvNet import ConvNet


class EmbedDataset(Dataset):
    """
    EmbedDataset:
    This dataset returns the precomputed embeddings and the final label when
    iterated over.
    """

    def __init__(self, embed_path: str, labels_path: str,
                 device: torch.device):
        self.device = device
        self.embeddings = utils.load_embeddings(embed_path)
        self.labels_dict = utils.load_labels(labels_path)
        self.headers = tuple(self.embeddings.keys())
        # Header refers to uniprot id
        assert len(self.headers) == len(self.labels_dict.keys())

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # given an index, return embedding and label sequence
        header = self.headers[index]
        embedding = torch.tensor(self.embeddings[header]).to(self.device)
        label = self.labels_dict[header]
        return embedding, label

    def __len__(self) -> int:
        return len(self.headers)


def get_dataloader(embed_path: str, labels_path: str,
                   batch_size: int, device: torch.device,
                   seed: int) -> DataLoader:
    torch.manual_seed(seed)
    dataset = EmbedDataset(embed_path=embed_path,
                           labels_path=labels_path,
                           device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def main_training_loop(model, device):
    bs = 20
    lr = 0.01
    optimizer = torch.optim.AdamOptimizer
    epochs = 10
    for i in range(epochs):
        pass


def train(model: torch.nn.Module,
          train_data: DataLoader,
          optimizer: torch.optim.Optimizer):
    """
    do a train on a minibatch
    :param model:
    :param train_data:
    :param optimizer:
    :return:
    """

    model.train()
    optimizer.zero_grad()

    all_targets = list()
    total_loss = 0

    for i, batch in enumerate(train_data):
        emb, label = batch

        optimizer.zero_grad()
        # output of model, then compute_loss(out, label)
        # convert label to ids => what is output of model?

        out = model(emb)

        # convert label to machine readable.

        assert out.size() == label.size()  ## (batchsize, length_of_longest_seq, 3)

        # loss =


def validate():
    pass


if __name__ == "__main__":
    ## Determine device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ## Data loading
    train_embeds_path = ""
    val_embeds_path = ""

    train_labels_path = "train.jsonl"
    val_labels_path = "val.jsonl"

    ## Load model
    cnn = ConvNet()

    ## Load Dataset (train and validate)
    main_training_loop(cnn, device)
    ##
