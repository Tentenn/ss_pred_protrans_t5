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
from transformers import T5EncoderModel, T5Tokenizer
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
        embedding = self.embeddings[header].to(self.device)
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


def main_training_loop(model, train_data, device):
    bs = 4
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 1
    for i in range(epochs):
        train(model, train_data, optimizer)


def train(model: torch.nn.Module,
          train_data: DataLoader,
          optimizer):
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
        print("label: ", label)

        assert out.size() == label.size()  ## (batchsize, length_of_longest_seq, 3)

        # loss =


def validate():
    pass


if __name__ == "__main__":
    ## Determine device
    print("Determine device")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ## Data loading
    print("Data loading")
    train_embeds_path = "/content/drive/MyDrive/BachelorThesis/data/train.jsonl_embeddings.h5"
    val_embeds_path = "/content/drive/MyDrive/BachelorThesis/data/val.jsonl_embeddings.h5"

    train_labels_path = "data/train.jsonl"
    val_labels_path = "data/val.jsonl"

    # train_loader = get_dataloader(embed_path=train_embeds_path, 
    #   labels_path=train_labels_path, batch_size=40, device=device, seed=42)

    # val_loader = get_dataloader(embed_path=val_embeds_path, 
    #  labels_path=val_labels_path, batch_size=40, device=device, seed=42)

    ### Test loader
    test_embeds_path = "/content/drive/MyDrive/BachelorThesis/data/new_pisces.jsonl_embeddings.h5"
    test_labels_path = "data/new_pisces.jsonl"
    test_loader = get_dataloader(embed_path=test_embeds_path, 
      labels_path=test_labels_path, batch_size=40, device=device, seed=42)

    ## Load model
    print("load Model")
    cnn = ConvNet()

    ## Load Dataset (train and validate)
    print("start Training")
    # main_training_loop(model=cnn, train_data=train_loader, device=device)
    ##
