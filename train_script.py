
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
from ConvNet2 import ConvNet2 # Old convnet

gc.enable()

"""
This code trains the CNN for 3-State secondary structure prediction
Using ProtTrans T5 per residue embeddings.
"""

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
  """
  # https://python.plainenglish.io/understanding-collate-fn-in-pytorch-f9d1742647d3
  # data is a list of batch size containing 2-tuple 
  # containing embedding and label seq
  """

  inputs = [torch.tensor(d[0]) for d in data] # converting embeds to tensor
  inputs = pad_sequence(inputs, batch_first=True) # pad to longest batch
  
  labels = [d[1] for d in data] # TODO: convert seq to (seqlen, 3) 

  return inputs, labels

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
  # add padding
  padded = [list(pad(subl, max_len, [0, 0, 0])) for subl in processed]
  return torch.tensor(np.array(padded), dtype=torch.float)


def main_training_loop(model: torch.nn.Module, 
                       train_data: DataLoader, 
                       val_data: DataLoader, 
                       device):
    bs = 80
    lr = 0.003
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 8

    # wandb logging
    config = {"learning_rate": lr,
              "epochs": epochs,
              "batch_size": bs,
              "optimizer": optimizer}
    wandb.init(project="emb_fex", entity="kyttang", config=config)
    # track best scores
    best_accuracy = float('-inf')
    best_loss = float('-inf')

    for epoch in range(epochs):
      # train model
      t_loss = train(model, train_data, optimizer)

      # validate results
      q3_accuracy, v_loss = validate(model, val_data)
      wandb.log({"accuracy (Q3)":q3_accuracy})
      
      # save model if better
      if q3_accuracy > best_accuracy:
        
        EPOCH = 5
        PATH = f"{bs}_{lr}_{epochs}_{round(q3_accuracy, 1)}_{t_loss}_cnn.pt"
        LOSS = 0.4

        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': t_loss,
                    }, PATH)


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

        emb, label = batch
        optimizer.zero_grad()
        out = model(emb)

        labels = process_label(label)

        loss = loss_fn(out, labels)
        loss.backward()
        
        losses.append(loss.item())
        wandb.log({"train_loss":loss.item()}) # logs loss for each batch

        optimizer.step()
    return sum(losses)/len(losses)

def validate(model: torch.nn.Module,
          val_data: DataLoader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    for i, batch in enumerate(val_data):
      emb, label = batch
      out = model(emb)
      acc_scores = []

      labels = process_label(label)
      loss = loss_fn(out, labels)
      wandb.log({"val_loss":loss.item()})

      for batch_idx, out_logits in enumerate(out):
        # Calculate scores for each sequence individually
        # And average over them

        seqlen = len(label[batch_idx])
        preds = logits_to_preds(out_logits[:seqlen]) # already in form: [0, 1, 2, 3]
        true_label = label_to_id(label[batch_idx]) # convert label to machine readable.

        acc = q3_acc(true_label, preds)
        acc_scores.append(acc)
      
    return sum(acc_scores)/len(acc_scores), np.std(acc_scores)

def test(model: torch.nn.Module,
          test_data: DataLoader,
         verbose=False):
    """
    verbose argument: whether or not to show actual predictions
    """
    model.eval()
    for i, batch in enumerate(test_data):
      emb, label = batch
      out = model(emb)
      acc_scores = []
      for batch_idx, out_logits in enumerate(out):
        # Calculate scores for each sequence individually
        # And average over them

        seqlen = len(label[batch_idx])
        preds = logits_to_preds(out_logits[:seqlen]) # already in form: [0, 1, 2, 3]
        true_label = label_to_id(label[batch_idx]) # convert label to machine readable.

        acc = q3_acc(true_label, preds)
        acc_scores.append(acc)

        if verbose:
          print(f"prediction:\t", preds_to_seq(preds))
          print(f"true label:\t", label[batch_idx])
          print()
      
    return sum(acc_scores)/len(acc_scores), np.std(acc_scores)

def preds_to_seq(preds):
  class_dict = {0:"H",1:"E",2:"C"}
  return "".join([class_dict[c.item()] for c in preds.reshape(-1)])

def q3_acc(y_true, y_pred):
  return accuracy_score(y_true, y_pred)

def sov(y_true, y_pred):
  pass

def load_sec_struct_model(path: str, model_type: torch.nn.Module):
  checkpoint_dir=path
  state = torch.load(checkpoint_dir, map_location=torch.device('cpu'))
  model = model_type
  if type(model_type) == ConvNet2:
    model.load_state_dict(state['state_dict'])
  elif type(model_type) == ConvNet:
    model.load_state_dict(state['model_state_dict'])
  else:
    raise Exception
  model = model.eval()
  model = model.to(device)
  # print('Loaded sec. struct. model from epoch: {:.1f}'.format(state['epoch']))

  return model

def predict_test_set():
  ## Test loader
  model_my = load_sec_struct_model("/content/drive/MyDrive/BachelorThesis/models/80_0.003_8_0.9_432.8474462689686_cnn.pt", ConvNet())


  # Predict on CASP12
  test_embeds_path = "/content/drive/MyDrive/BachelorThesis/data/casp12.jsonl_embeddings.h5"
  test_labels_path = "data/casp12.jsonl"
  test_loader = get_dataloader(embed_path=test_embeds_path, 
                                labels_path=test_labels_path, 
                                batch_size=4, device=device, seed=42)

  acc, std = test(model_my, test_loader, verbose=True)
  print("CASP12", acc, std)



  # Predict on new_pisces
  test2_embeds_path = "/content/drive/MyDrive/BachelorThesis/data/new_pisces.jsonl_embeddings.h5"
  test2_labels_path = "data/new_pisces.jsonl"
  test2_loader = get_dataloader(embed_path=test2_embeds_path, 
                                labels_path=test2_labels_path, 
                                batch_size=4, device=device, seed=42)

  acc, std = test(model_my, test2_loader, verbose=True)
  print("NEW_PISCES", acc, std)

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
    
    ## Test on the available test_sets
    # predict_test_set()