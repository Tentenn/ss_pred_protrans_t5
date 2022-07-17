import torch
from torch import nn
import torch.optim as optim
from typing import Any
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
from datetime import datetime
import random
import argparse


import utils
from Dataset import SequenceDataset
from T5ConvNet import T5CNN
from smart_optim import Adamax
from transformers import Adafactor


"""
This code trains the CNN for 3-State secondary structure prediction
Using ProtTrans T5 per residue embeddings.
"""

def process_labels(labels: list, mask:list, onehot=True):
  """
  turns a list of labels ['HEECCC', 'HHHEEEECCC'] to one hot encoded tensor
  and add padding e.g torch.tensor([[1, 0, 0], [0, 0, 1], [0, 0, 0]])
  """

  max_len = len(max(labels, key=len)) # determine longest sequence in list
  processed = []
  if onehot:
    class_mapping = {"H":[1, 0, 0], "E":[0, 1, 0], "L":[0, 0, 1], "C":[0, 0, 1]}
    processed = [[class_mapping[c] for c in label] for label in labels]
    # add padding manually using [0, 0, 0]
    processed = [list(pad(subl, max_len, [0, 0, 0])) for subl in processed]
  else:
    class_mapping = {"H":0, "E":1, "L":2, "C":2}
    processed = [[class_mapping[c] for c in label] for label in labels]
    # add mask
    for i,e in enumerate(mask):
      pel = [-1 if e[j]==0 else p for j,p in enumerate(processed[i])]
      processed[i] = pel
    # add padding
    processed = [list(pad(subl, max_len, -1)) for subl in processed]
    return torch.tensor(np.array(processed), dtype=torch.long)


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

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

def process_label(labels: list):
  """
  turns a list of labels ['HEECCC', 'HHHEEEECCC'] to one hot encoded tensor
  and add padding e.g torch.tensor([[1, 0, 0], [0, 0, 1], [0, 0, 0]])
  """

  max_len = len(max(labels, key=len))
  class_mapping = {"H":[1, 0, 0], "E":[0, 1, 0], "L":[0, 0, 1], "C":[0, 0, 1]}
  processed = [[class_mapping[c] for c in label] for label in labels]
  # add padding manually using [0, 0, 0]
  padded = [list(pad(subl, max_len, [0, 0, 0])) for subl in processed]
  return torch.tensor(np.array(padded), dtype=torch.float)

def test(model: torch.nn.Module,
          test_data: DataLoader,
         verbose=False):
    """
    verbose argument: whether or not to show actual predictions
    """
    model.eval()
    acc_scores = []
    for i, batch in enumerate(test_data):
      ids, label, mask = batch

      labels_f = process_labels(label, mask=mask, onehot=False).to(device)
      ids = ids.to(device)
      mask = mask.to(device)

      with torch.no_grad():
        out = model(ids)
      for batch_idx, out_logits in enumerate(out):
        # Calculate scores for each sequence individually
        # And average over them

        seqlen = len(label[batch_idx])
        preds = logits_to_preds(out_logits[:seqlen]) # already in form: [0, 1, 2, 3
        true_label = label_to_id(label[batch_idx]) # convert label to machine readable.
        res_mask = mask[batch_idx][:seqlen] # [:seqlen] to cut the padding

        assert seqlen == len(preds) == len(res_mask), "length of seqs not matching"
        acc = q3_acc(true_label, preds, res_mask)
        acc_scores.append(acc)

        if verbose:
          print(f"prediction:\t", preds_to_seq(preds))
          print(f"true label:\t", label[batch_idx])
          print("accuracy:\t", acc)
          print()
      
    return sum(acc_scores)/len(acc_scores), np.std(acc_scores)

def preds_to_seq(preds):
  class_dict = {0:"H",1:"E",2:"C"}
  return "".join([class_dict[c.item()] for c in preds.reshape(-1)])

def q3_acc(y_true, y_pred, mask):
  return accuracy_score(y_true, y_pred, sample_weight=[int(e) for e in mask])

def sov(y_true, y_pred):
  pass


def custom_collate(data):
      """
      # https://python.plainenglish.io/understanding-collate-fn-in-pytorch-f9d1742647d3
      # data is a list of len batch size containing 3-tuple 
      # containing seq, labels and mask
      """

      inputs = [torch.tensor(d[0]) for d in data] # converting embeds to tensor
      # inputs = [d[0] for d in data]
      inputs = pad_sequence(inputs, batch_first=True) # pad to longest batch

      # now = datetime.now()
      # current_time = now.strftime("%H:%M:%S")
      # print(f"[{current_time}] shape", inputs.shape)
      
      labels = [d[1] for d in data]
      res_mask = [torch.tensor([float(dig) for dig in d[2]]) for d in data]
      mask = pad_sequence(res_mask, batch_first=True)
      
      
      return inputs, labels, mask

def seq_collate(data):
  """
  # https://python.plainenglish.io/understanding-collate-fn-in-pytorch-f9d1742647d3
  # data is a list of len batch size containing 3-tuple 
  # containing seq, labels and mask
  """

  inputs = [torch.tensor(d[0]) for d in data] # converting embeds to tensor
  inputs = pad_sequence(inputs, batch_first=True) # pad to longest batch

  # now = datetime.now()
  # current_time = now.strftime("%H:%M:%S")
  # print(f"[{current_time}] shape", inputs.shape)
  # print(inputs)
  
  labels = [d[1] for d in data]
  res_mask = [torch.tensor([float(dig) for dig in d[2]]) for d in data]
  mask = pad_sequence(res_mask, batch_first=True)
  
  return inputs, labels, mask


def get_dataloader(jsonl_path: str, batch_size: int, device: torch.device,
                   seed: int, max_emb_size: int) -> DataLoader:
    torch.manual_seed(seed)
    dataset = SequenceDataset(jsonl_path=jsonl_path,
                           device=device,
                           max_emb_size=max_emb_size)
    loader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=True, collate_fn=custom_collate)
    return loader

if __name__ == "__main__":
    import sys
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using", device)
    
    ## Data loading
    drive_path = "/home/ubuntu/instance1/data/"

    # Test loader
    casp12_path = drive_path + "casp12.jsonl"
    casp12_loader = get_dataloader(jsonl_path=casp12_path, batch_size=1, device=device, seed=42,
                                 max_emb_size=5000)

    npis_path = drive_path + "new_pisces.jsonl"
    npis_loader = get_dataloader(jsonl_path=npis_path, batch_size=1, device=device, seed=42,
                                 max_emb_size=5000)
    
    ## Test data (TODO)
                        
    ## Load model
    model = T5CNN()
    model.load_state_dict(torch.load(sys.argv[1]))
    model = model.to(device)
    print("new_pisces:", test(model, npis_loader, verbose=False))
    print("casp12:", test(model, casp12_loader, verbose=True))
    
    