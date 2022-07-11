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

import utils
from Dataset import SequenceDataset
from T5ConvNet import T5CNN
from ProtBertCcnn import ProtBertCNN

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


def main_training_loop(model: torch.nn.Module, 
                       train_data: DataLoader, 
                       val_data: DataLoader,
                       batch_size: int, 
                       lr: float,
                       epochs: int,
                       grad_accum: int,
                       optimizer_name: str,
                       loss_fn,
                       device):
    
    if optimizer_name == "adam":
      optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
      assert False, f"Optimizer {optimizer_name} not implemented"
    # track best scores
    best_accuracy = 0.87 # float('-inf')
    # best_loss = float('-inf')

    for epoch in range(epochs):
      # train model and save train loss
      print(f"train epoch {epoch}")
      t_loss = train(model, train_data, loss_fn, optimizer, grad_accum)

      # validate results and calculate scores
      q3_accuracy, v_loss = validate(model, val_data, loss_fn)
      wandb.log({"accuracy (Q3)":q3_accuracy})
      wandb.log({"val_loss":v_loss})
      
      # save model if better
      if q3_accuracy > best_accuracy:
        best_accuracy = q3_accuracy
        PATH = f"{batch_size}_{lr}_{epochs}_{round(q3_accuracy, 1)}_{t_loss}_cnn.pt"
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': t_loss,
                    }, PATH)


def train(model: torch.nn.Module,
          train_data: DataLoader,
          loss_fn,
          optimizer,
          grad_accum):
    """
    do a train on a minibatch
    """
    gc.collect()
    model.train()
    # optimizer.zero_grad()
    total_loss = 0
    count = 0
    # batch accumulation parameter
    accum_iter = grad_accum
    for i, batch in enumerate(train_data):
        print(f"batch-{i}")
        ids, label, mask = batch
        ids = ids.to(device)
        mask = mask.to(device)
        # passes and weights update
        with torch.set_grad_enabled(True):
          out = model(ids) # shape: [bs, max_seq_len, 3]
          # string to float conversion, padding and mask labels
          labels = process_labels(label, mask=mask, onehot=False).to(device)
          # reshape to make loss work 
          out = torch.transpose(out, 1, 2)
          assert out.shape[-1] == labels.shape[-1], f"out: {out.shape}, labels: {labels.shape}"
          loss = loss_fn(out, labels)#  / accum_iter
          loss.backward()
          total_loss += loss.item()
          count += 1
          wandb.log({"train_loss":loss.item()}) # logs loss for each batch

          # weights update
          if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_data)):
            print(f"update")
            optimizer.step()
            optimizer.zero_grad()
            print("update done")
    return total_loss/count

def validate(model: torch.nn.Module,
          val_data: DataLoader,
          loss_fn):
    model.eval()
    
    last_accuracy = 0
    total_loss = 0
    count = 0
    for i, batch in enumerate(val_data):

      ids, label, mask = batch

      labels_f = process_labels(label, mask=mask, onehot=False).to(device)
      ids = ids.to(device)
      mask = mask.to(device)

      with torch.no_grad():
        out = model(ids)


      # reshape to make loss work 
      out_f = torch.transpose(out, 1, 2)

      # calculate loss

      loss = loss_fn(out_f, labels_f)
      total_loss += loss
      count += 1
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

    return last_accuracy, total_loss/count

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
      # inputs = [d[0] for d in data]
      inputs = pad_sequence(inputs, batch_first=True) # pad to longest batch

      now = datetime.now()
      current_time = now.strftime("%H:%M:%S")
      print(f"[{current_time}] shape", inputs.shape)
      
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

  now = datetime.now()
  current_time = now.strftime("%H:%M:%S")
  print(f"[{current_time}] shape", inputs.shape)
  print(inputs)
  
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
    batch_size = 2
    grad_accum = 10
    max_emb_size = 305
    optimizer_name = "adam"
    lr = 0.0001
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    epochs = 1
    model_type = "pt5-cnn"
    seed = 42

    ## Determine device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ## Data loading
    drive_path = "/content/drive/MyDrive/BachelorThesis/data/"
    train_path = drive_path + "train_200.jsonl"
    val_path = drive_path + "val_200.jsonl"

    # train_loader = get_dataloader(jsonl_path=train_path, 
    #                               batch_size=batch_size, 
    #                               device=device, seed=42,
    #                               max_emb_size=max_emb_size)

    # val_loader = get_dataloader(jsonl_path=val_path, 
    #                             batch_size=batch_size, 
    #                             device=device, seed=42,
    #                             max_emb_size=max_emb_size)

    # Test loader
    casp12_path = drive_path + "casp12_300.jsonl"
    casp12_loader = get_dataloader(jsonl_path=casp12_path, batch_size=batch_size, device=device, seed=seed,
                                 max_emb_size=max_emb_size)

    npis_path = drive_path + "new_pisces_300.jsonl"
    npis_loader = get_dataloader(jsonl_path=npis_path, batch_size=batch_size, device=device, seed=seed,
                                 max_emb_size=max_emb_size)

    # Chose model
    if model_type == "pt5-cnn":
      model = T5CNN().to(device)
    elif model_type == "pbert-cnn":
      model = ProtBertCNN().to(device)
    else:
      assert False, f"Model type not implemented {model_type}"

    # For testing and logging
    train_data = npis_loader
    val_data = casp12_loader

    # wandb logging
    config = {"lr": str(lr).replace("0.", ""),
              "epochs": epochs,
              "batch_size": batch_size,
              "grad_accum": grad_accum,
              "optim_name": optimizer_name,
              "model_type": model_type,
              "loss_fn": loss_fn,
              "train_path": train_path,
              "val_path": val_path,
              "casp12_path": casp12_path,
              "npis_path": npis_path,
              "train_size": len(train_data),
              "val_size": len(val_data)
              }
    experiment_name = f"{model_type}-{batch_size}_{lr}_{epochs}_{grad_accum}"
    wandb.init(project="t5cnn-ft", entity="kyttang", config=config, name=experiment_name)

    # start training
    gc.collect()
    main_training_loop(model=model, 
                        train_data=train_data, 
                        val_data=val_data, 
                        device=device, 
                        batch_size=batch_size,
                        lr=lr,
                        epochs=epochs,
                        grad_accum=grad_accum,
                        optimizer_name=optimizer_name,
                        loss_fn=loss_fn)
    
    ## Test data (TODO)