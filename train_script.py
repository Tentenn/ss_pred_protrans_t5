
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
from torch.nn.utils.rnn import pad_sequence
from itertools import chain, repeat, islice
from sklearn.metrics import accuracy_score
import gc
import wandb

import utils
from ConvNet import ConvNet

gc.enable()

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
    loader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=True, collate_fn=custom_collate)
    return loader

def logits_to_preds(logits):
  """
  @param logits: a tensor of size (seqlen, 3) containing logits of ss preds
  @returns: a list of predictions eg. [2, 1, 1, 2, 2, 2, 0] 
  => dssp3 class_mapping = {0:"H",1:"E",2:"L"} 
  """
  preds = torch.max(logits, dim=1 )[1].detach().cpu().numpy().squeeze()
  return preds

def label_to_id(labels: str):
  """
  'HHEELLL' -> [0, 0, 1, 1, 2, 2, 2]
  """
  class_mapping = {"H":0, "E":1, "L":2, "C":2} 
  converted = [[class_mapping[c] for c in label] for label in labels]
  return torch.tensor(converted)

def custom_collate(data):
  # https://python.plainenglish.io/understanding-collate-fn-in-pytorch-f9d1742647d3

  # data is a list of batch size containing 2-tuple containing embedding and label seq

  inputs = [torch.tensor(d[0]) for d in data] # converting embeds to tensor
  inputs = pad_sequence(inputs, batch_first=True) # pad embeddings to longest batch
  
  labels = [d[1] for d in data] # TODO: convert seq to (seqlen, 3) 

  return inputs, labels

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

def process_label(labels: list):
  # a list of labels ['HEECCC', 'HHHEEEECCC']
  max_len = len(max(labels, key=len))
  class_mapping = {"H":[1, 0, 0], "E":[0, 1, 0], "L":[0, 0, 1], "C":[0, 0, 1]}
  processed = [[class_mapping[c] for c in label] for label in labels]
  # add padding
  padded = [list(pad(subl, max_len, [0, 0, 0])) for subl in processed]
  return torch.tensor(np.array(padded), dtype=torch.float)


def main_training_loop(model, train_data, val_data, device):
    bs = 80
    lr = 0.002
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 15

    config = {"learning_rate": lr,
              "epochs": epochs,
              "batch_size": bs,
              "optimizer": optimizer}
    wandb.init(project="emb_fex", entity="kyttang", config=config)
    

    for i in range(epochs):
      # train
      t_loss = train(model, train_data, optimizer)
      # wandb.log({"train_loss":t_loss})
      # print("train_loss: ", t_loss)

      # validate results
      q3_accuracy, v_loss = validate(model, val_data)
      # print("accuracy: ", q3_accuracy)
      wandb.log({"accuracy (Q3)":q3_accuracy})





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
    loss_fn = nn.CrossEntropyLoss()

    all_targets = list()
    losses = []

    for i, batch in enumerate(train_data):
        batch_loss = 0
        emb, label = batch
        optimizer.zero_grad()
        out = model(emb)

        labels = process_label(label)

        loss = loss_fn(out, labels)
        loss.backward()
        
        # print(loss.item())
        losses.append(loss.item())
        wandb.log({"train_loss":loss.item()})

        optimizer.step()

        # print("out: ", out) # out.shape: torch.Size(bs, max_batch_seq_len, 3)
        # calculate loss for each sequence in the batch
        # for batch_idx, out_logits in enumerate(out): ## Geht das nicht auch schneller?
        #   # 'e' are 3 class logits of a sequence
        #   optimizer.zero_grad()

        #   seqlen = len(label[batch_idx])
        #   # preds = logits_to_preds(out_logits[:seqlen]) # already in form: [0, 1, 2, 3]
        #   true_label = label_to_id(label[batch_idx]) # convert label to machine readable.

        #   # calculate loss for a sequece
        #   # assert out_logits.size()[0] == true_label.size()
        #   loss = loss_fn(out_logits[:seqlen], true_label)
        #   loss.backward()
        #   print("loss: ", loss.item())
        #   optimizer.step()
    return sum(losses)/len(losses)

def validate(model: torch.nn.Module,
          train_data: DataLoader):
    model.eval()
    for i, batch in enumerate(train_data):
      emb, label = batch
      out = model(emb)
      acc_scores = []

      for batch_idx, out_logits in enumerate(out): ## Geht das nicht auch schneller?

          seqlen = len(label[batch_idx])
          preds = logits_to_preds(out_logits[:seqlen]) # already in form: [0, 1, 2, 3]
          true_label = label_to_id(label[batch_idx]) # convert label to machine readable.

          acc = q3_acc(true_label, preds)
          acc_scores.append(acc)
      
      
      return sum(acc_scores)/len(acc_scores), -1

def q3_acc(y_true, y_pred):
  return accuracy_score(y_true, y_pred)

def sov(y_true, y_pred):
  pass

if __name__ == "__main__":
    ## Collect garbage
    gc.collect()

    ## Determine device
    print("Determine device")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ## Data loading
    print("Data loading")
    train_embeds_path = "/content/drive/MyDrive/BachelorThesis/data/train.jsonl_embeddings.h5"
    val_embeds_path = "/content/drive/MyDrive/BachelorThesis/data/val.jsonl_embeddings.h5"

    train_labels_path = "data/train.jsonl"
    val_labels_path = "data/val.jsonl"

    train_loader = get_dataloader(embed_path=train_embeds_path, 
      labels_path=train_labels_path, batch_size=40, device=device, seed=42)

    val_loader = get_dataloader(embed_path=val_embeds_path, 
     labels_path=val_labels_path, batch_size=40, device=device, seed=42)

    ### Test loader
    # test_embeds_path = "/content/drive/MyDrive/BachelorThesis/data/new_pisces.jsonl_embeddings.h5"
    # test_labels_path = "data/new_pisces.jsonl"
    # test_loader = get_dataloader(embed_path=test_embeds_path, 
    #   labels_path=test_labels_path, batch_size=4, device=device, seed=42)
    # ###

    ## Load model
    print("load Model")
    cnn = ConvNet()

    ## Train and validate (train and validate)
    print("start Training")
    main_training_loop(model=cnn, train_data=train_loader, val_data=val_loader, device=device)
    ##
