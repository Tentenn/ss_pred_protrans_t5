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

import utils
from Dataset import SequenceDataset
from T5ConvNet import T5CNN
from ProtBertCcnn import ProtBertCNN

"""
This code trains the CNN for 3-State secondary structure prediction
Using ProtTrans T5 per residue embeddings.
"""


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


def main_training_loop(model: torch.nn.Module, 
                       train_data: DataLoader, 
                       val_data: DataLoader,
                       batch_size: int, 
                       device):
    batch_size = 4
    lr = 0.003
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 1

    

    # wandb logging
    config = {"learning_rate": lr,
              "epochs": epochs,
              "batch_size": batch_size,
              "optimizer": optimizer}
    wandb.init(project="t5cnn-ft", entity="kyttang", config=config)
    # track best scores
    best_accuracy = float('-inf')
    # best_loss = float('-inf')

    for epoch in range(epochs):
      # train model and save train loss
      t_loss = train(model, train_data, loss_fn, optimizer)

      # validate results and calculate scores
      q3_accuracy, v_loss = validate(model, val_data, loss_fn)
      wandb.log({"accuracy (Q3)":q3_accuracy})
      wandb.log({"val_loss":v_loss})
      
      # save model if better
      if q3_accuracy > best_accuracy:
        best_accuracy = q3_accuracy
        PATH = f"{bs}_{lr}_{epochs}_{round(q3_accuracy, 1)}_{t_loss}_cnn.pt"
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': t_loss,
                    }, PATH)


def train(model: torch.nn.Module,
          train_data: DataLoader,
          loss_fn,
          optimizer):
    """
    do a train on a minibatch
    
    Experimentell übergeben wir die resolution mask 
    während dem embeddings generieren
    """

    model.train()
    optimizer.zero_grad()
    

    losses = []

    for i, batch in enumerate(train_data):
        ids, label, mask = batch

        ids = ids.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        out = model(ids, mask) # shape: [bs, max_seq_len, 3]

        labels = process_label(label).to(device)

        # mask out disordered aas
        out = out * mask.unsqueeze(-1)
        labels = labels * mask.unsqueeze(-1)

        # remove zero tensors from 2nd dim
        nonZeroRows = torch.abs(out).sum(dim=2) > 0

        loss = loss_fn(out[nonZeroRows], labels[nonZeroRows])
        loss.backward()
        
        losses.append(loss.item())
        wandb.log({"train_loss":loss.item()}) # logs loss for each batch

        optimizer.step()
    return sum(losses)/len(losses)

def validate(model: torch.nn.Module,
          val_data: DataLoader,
          loss_fn):
    model.eval()
    
    last_accuracy = 0
    losses = []
    for i, batch in enumerate(val_data):
      ids, label, mask = batch
      out = model(ids, mask)
      labels = process_label(label).to(device)
      ids = ids.to(device)
      mask = mask.to(device)

      # mask out disordered aas
      out = out * mask.unsqueeze(-1)
      labels = labels * mask.unsqueeze(-1)

      # remove padding and disordered aas (0-vectors)
      nonZeroRows = torch.abs(out).sum(dim=2) > 0

      # calculate loss
      loss = loss_fn(out[nonZeroRows], labels[nonZeroRows])
      losses.append(loss)
      # wandb.log({"val_loss":loss.item()})

      acc_scores = []
      for batch_idx, out_logits in enumerate(out):
        # Calculate scores for each sequence individually
        # And average over them

        seqlen = len(label[batch_idx])
        preds = logits_to_preds(out_logits[:seqlen]) # already in form: [0, 1, 2, 3]
        true_label = label_to_id(label[batch_idx]) # convert label to machine readable.
        res_mask = mask[batch_idx][:seqlen] # [:seqlen] to cut the padding

        assert seqlen == len(preds) == len(res_mask), "length of seqs not matching"
        
        acc = q3_acc(true_label, preds, res_mask)
        acc_scores.append(acc)
      last_accuracy = sum(acc_scores)/len(acc_scores)# , np.std(acc_scores)

    return last_accuracy, sum(losses)/len(losses)

def test(model: torch.nn.Module,
          test_data: DataLoader,
         verbose=False):
    """
    verbose argument: whether or not to show actual predictions
    """
    model.eval()
    acc_scores = []
    for i, batch in enumerate(test_data):
      emb, label, mask = batch
      out = model(emb)
      for batch_idx, out_logits in enumerate(out):
        # Calculate scores for each sequence individually
        # And average over them

        seqlen = len(label[batch_idx])
        preds = logits_to_preds(out_logits[:seqlen]).cpu() # already in form: [0, 1, 2, 3]
        true_label = label_to_id(label[batch_idx]).cpu() # convert label to machine readable.
        res_mask = mask[batch_idx][:seqlen].cpu() # [:seqlen] to cut the padding

        assert seqlen == len(preds) == len(res_mask), "length of seqs not matching"
        acc = q3_acc(true_label, preds, res_mask)
        acc_scores.append(acc)

        if verbose:
          print(f"prediction:\t", preds_to_seq(preds))
          print(f"true label:\t", label[batch_idx])
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
      inputs = pad_sequence(inputs, batch_first=True) # pad to longest batch

      print("shape", inputs.shape)
      
      labels = [d[1] for d in data]
      res_mask = [torch.tensor([float(dig) for dig in d[2]]) for d in data]
      mask = pad_sequence(res_mask, batch_first=True)
      
      
      return inputs, labels, mask

def get_dataloader(jsonl_path: str, batch_size: int, device: torch.device,
                   seed: int) -> DataLoader:
    torch.manual_seed(seed)
    dataset = SequenceDataset(jsonl_path=jsonl_path,
                           device=device)
    loader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=True, collate_fn=custom_collate)
    return loader

if __name__ == "__main__":
    ## Collect garbage
    gc.collect()
    batch_size = 1

    ## Determine device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ## Data loading
    print("(1) Data loading...", end="")
    drive_path = "/content/drive/MyDrive/BachelorThesis/data/"
    train_path = drive_path + "train_400.jsonl"
    val_path = drive_path + "val_400.jsonl"

    # train_loader = get_dataloader(jsonl_path=train_path, batch_size=40, device=device, seed=42)

    # val_loader = get_dataloader(jsonl_path=val_path, batch_size=40, device=device, seed=42)

    ## Test loader
    casp12_path = drive_path + "casp12_400.jsonl"
    casp12_loader = get_dataloader(jsonl_path=casp12_path, batch_size=batch_size, device=device, seed=42)

    npis_path = drive_path + "new_pisces_400.jsonl"
    npis_loader = get_dataloader(jsonl_path=npis_path, batch_size=batch_size, device=device, seed=42)
    ##


    ## Load model
    print("Check! \n (2) load Model...", end="")
    model = ProtBertCNN().to(device)

    ## Train and validate (train and validate)
    print("Check! \n (3) start Training.. ")
    main_training_loop(model=model, train_data=npis_loader, val_data=casp12_loader, device=device, batch_size=batch_size)
    
    ## 