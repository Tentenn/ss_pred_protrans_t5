
"""
This code trains the CNN for 3-State secondary structure prediction
Using ProtTrans T5 per residue embeddings.
"""
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
import sys
import random
import argparse

import utils
from ConvNet import ConvNet
from Dataset import EmbedDataset

"""
This code trains the CNN for 3-State secondary structure prediction
Using ProtTrans T5 per residue embeddings.
"""

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
  return torch.tensor(converted).squeeze(-1)

def custom_collate(data):
  """
  # https://python.plainenglish.io/understanding-collate-fn-in-pytorch-f9d1742647d3
  # data is a list of batch size containing 3-tuple 
  # containing embedding and label seq
  """

  inputs = [d[0] for d in data] # converting embeds to tensor
  inputs = pad_sequence(inputs, batch_first=True) # pad to longest batch
  
  labels = [d[1] for d in data]
  res_mask = [torch.tensor([float(dig) for dig in d[2]]) for d in data]
  mask = pad_sequence(res_mask, batch_first=True)

  return inputs, labels, mask

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

def process_label(labels: list, mask:list, onehot=True):
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


def main_training_loop(model: torch.nn.Module, 
                       train_data: DataLoader, 
                       val_data: DataLoader, 
                       device,
                       args):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # wandb logging
    config = {"learning_rate": args.lr,
              "epochs": args.epochs,
              "batch_size": args.bs,
              "optimizer": optimizer}
    
    exp_name = f"{args.wname}_{random.randint(300, 999)}_lr={args.lr}_ep={args.epochs}_bs={args.bs}"
    wandb.init(project=args.pname, entity="kyttang", config=config, name=exp_name)
    # track best scores
    best_accuracy = float('-inf')
    best_loss = float('-inf')

    for epoch in range(args.epochs):
      # train model and save train loss
      t_loss = train(model, train_data, optimizer)

      # validate results and calculate scores
      q3_accuracy, v_loss, std = validate(model, val_data)
      wandb.log({"accuracy (Q3)":q3_accuracy})
      wandb.log({"val_loss":v_loss})
      wandb.log({"val_std":std})
      
      # save model if better
      if q3_accuracy > best_accuracy:
        best_accuracy = q3_accuracy
        # PATH = f"bs={bs}_lr={lr}_te={epoch}_{round(q3_accuracy, 3)}_{t_loss}_cnn.pt"
        # torch.save(model.state_dict(), PATH)
        print("[DEV] Not saving models")


def train(model: torch.nn.Module,
          train_data: DataLoader,
          optimizer):
    """
    do a train on a minibatch
    """

    model.train()
    optimizer.zero_grad()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    losses = []

    for i, batch in enumerate(train_data):
        emb, label, mask = batch
        optimizer.zero_grad()
        emb = emb.to(device)
        out = model(emb) # shape: [bs, max_seq_len, 3]

        # string to float conversion, padding and mask labels
        labels = process_label(label, mask=mask, onehot=False).to(device)

        # reshape to make loss work 
        out = torch.transpose(out, 1, 2).to(device)

        # # mask out disordered aas
        # out = out * mask.unsqueeze(-1)
        # labels = labels * mask.unsqueeze(-1)

        # # remove zero tensors from 2nd dim
        # nonZeroRows = torch.abs(out).sum(dim=2) > 0
        # out = out[nonZeroRows]
        # labels = labels[nonZeroRows]
        # # Experimental

        loss = loss_fn(out, labels)
        loss.backward()
        
        losses.append(loss.item())
        wandb.log({"train_loss":loss.item()}) # logs loss for each batch

        optimizer.step()
    return sum(losses)/len(losses)

def validate(model: torch.nn.Module,
          val_data: DataLoader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    
    last_accuracy = 0
    losses = []
    acc_scores = []
    for i, batch in enumerate(val_data):
      emb, label, mask = batch
      emb = emb.to(device)
      out = model(emb).to(device) # shape: [bs, max_seq_len, 3]
      # string to float conversion
      labels_f = process_label(label, mask=mask, onehot=False).to(device)

      # reshape to make loss work 
      max_batch_len = len(labels_f[0])
      bs = len(label)
      out_f = torch.transpose(out, 1, 2).to(device)

      # calculate loss, ignores -1 elements (padding and masking)
      loss = loss_fn(out_f, labels_f)
      losses.append(loss)
      # wandb.log({"val_loss":loss.item()})

      
      for batch_idx, out_logits in enumerate(out):
        # Calculate scores for each sequence individually
        # And average over them

        seqlen = len(label[batch_idx])
        preds = logits_to_preds(out_logits[:seqlen]) # already in form: [0, 1, 2, 3]
        true_label = label_to_id(label[batch_idx]) # convert label to machine readable.
        res_mask = mask[batch_idx][:seqlen] # [:seqlen] to cut the padding

        # print(seqlen, len(preds), len(res_mask))
        assert seqlen == len(preds) == len(res_mask), f"length of seqs not matching"
        
        acc = q3_acc(true_label, preds, res_mask)
        acc_scores.append(acc)
    last_accuracy = sum(acc_scores)/len(acc_scores)# , np.std(acc_scores)

    return last_accuracy, sum(losses)/len(losses), np.std(acc_scores)

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
      print(out.shape)
      for batch_idx, out_logits in enumerate(out):
        # Calculate scores for each sequence individually
        # And average over them

        seqlen = len(label[batch_idx])
        # print(out_logits)
        preds = logits_to_preds(out_logits[:seqlen]) # already in form: [0, 1, 2, 3]
        true_label = label_to_id(label[batch_idx]) # convert label to machine readable.
        res_mask = mask[batch_idx][:seqlen] # [:seqlen] to cut the padding

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
  # print("ytrue: ", y_true)
  # print("ypred: ", y_pred)
  return accuracy_score(y_true, y_pred, sample_weight=[int(e) for e in mask])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=4)
    # parser.add_argument("--lmt", type=str, default="bert", help="Choose language model type: pt5, pbert")
    parser.add_argument("--temb", type=str, help="path for training embeddings")
    parser.add_argument("--vemb", type=str, help="path for validation embeddings")
    parser.add_argument("--pname", type=str, help="project name for wandb")
    parser.add_argument("--wname", type=str, help="name of run")
    parser.add_argument("--train_labels", type=str, help="jsonl file", default="data/train.jsonl")
    args = parser.parse_args()

    ## Determine device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device}")

    ## Data loading
    print("Data loading")
    # train_embeds_path = "/content/drive/MyDrive/BachelorThesis/data/train.jsonl_embeddings.h5"
    # train_embeds_path = "/notebooks/ss_pred_protrans_t5/data/train.jsonl-Rostlab-prot_t5_xl_half_uniref50-enc-_pt5.h5"
    # val_embeds_path = "/content/drive/MyDrive/BachelorThesis/data/val.jsonl_embeddings.h5"
    # val_embeds_path = "/notebooks/ss_pred_protrans_t5/data/val.jsonl-Rostlab-prot_t5_xl_half_uniref50-enc-_pt5.h5"

    train_labels_path = args.train_labels
    val_labels_path = "data/val.jsonl"

    train_loader = get_dataloader(embed_path=args.temb, 
      labels_path=train_labels_path, batch_size=args.bs, device=device, seed=42)

    val_loader = get_dataloader(embed_path=args.vemb, 
     labels_path=val_labels_path, batch_size=args.bs, device=device, seed=42)

    ### Test loader
    # test_embeds_path = "/content/drive/MyDrive/BachelorThesis/data/new_pisces.jsonl_embeddings.h5"
    # test_labels_path = "data/new_pisces.jsonl"
    # test_loader = get_dataloader(embed_path=test_embeds_path, 
    #   labels_path=test_labels_path, batch_size=4, device=device, seed=42)
    ###

    ## Load model
    print("load Model")
    cnn = ConvNet()
    cnn = cnn.to(device)

    ## Train and validate (train and validate)
    print("start Training")
    main_training_loop(model=cnn, train_data=train_loader, val_data=val_loader, device=device, args=args)
    
    ## 