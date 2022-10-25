import torch
from torch import nn
import torch.optim as optim
from typing import Any
import json
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import T5EncoderModel, T5Tokenizer, BertTokenizer
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
from T5Linear import T5Linear
from smart_optim import Adamax
from transformers import Adafactor
from BertLinear import BertLinear
from BertCNN import BertCNN


"""
This code trains the CNN for 3-State secondary structure prediction
Using ProtTrans T5 per residue embeddings.
"""

CLASS_MAPPING = {"H":0, "E":1, "L":2, "C":2}

def process_labels(labels: list, mask:list, onehot=False):
  """
  turns a list of labels ['HEECCC', 'HHHEEEECCC'] labels and adds padding
  """
  max_len = len(max(labels, key=len)) # determine longest sequence in list
  processed = []
  processed = [[CLASS_MAPPING[c] for c in label] for label in labels]
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
  converted = [[CLASS_MAPPING[c] for c in label] for label in labels]
  return torch.tensor(converted)

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

def main_training_loop(model: torch.nn.Module, 
                       train_data: DataLoader, 
                       val_data: DataLoader,
                       batch_size: int, 
                       lr: float,
                       epochs: int,
                       grad_accum: int,
                       optimizer_name: str,
                       weight_decay: float,
                       loss_fn,
                       freeze_epoch: int,
                       device):
    
    if optimizer_name == "adam":
      optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "adamax":
      optimizer = Adamax(model.parameters(), lr=lr)
    elif optimizer_name == "adafactor":
      optimizer = Adafactor(model.parameters(), lr=lr, relative_step=False, scale_parameter=False, weight_decay=weight_decay)
    elif optimizer_name == "adafactor_rs":
      optimizer = Adafactor(model.parameters(), weight_decay=weight_decay)
    elif optimizer_name == "adagrad":
      optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == "rmsprop":
      optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
      assert False, f"Optimizer {optimizer_name} not implemented"
    # track best scores
    best_accuracy = float('-inf')
    # best_loss = float('-inf')
    
    epochs_without_improvement = 0
    best_vloss = 10000
    
    for epoch in range(epochs):
      # train model and save train loss
      print(f"train epoch {epoch}")
      t_loss = train(model, train_data, loss_fn, optimizer, grad_accum)
      print("t_loss:",t_loss)

      # validate results and calculate scores
      print(f"validate epoch {epoch}")
      q3_accuracy, v_loss, std = validate(model, val_data, loss_fn)
      wandb.log({"accuracy (Q3)":q3_accuracy})
      wandb.log({"val_loss":v_loss})
      wandb.log({"val_std":std})
      print("acc:", q3_accuracy)
    
      # update best vloss
      if v_loss < best_vloss: # smaller is better
        best_vloss = v_loss
        epochs_without_improvement = 0
      else:
        epochs_without_improvement += 1
        print(f"Epochs without improvement: {epochs_without_improvement}")
        if epochs_without_improvement >= 2:
            print("max amount of epochs without improvement reached. Stopping training...")
            break
      
      # save model if better
      if q3_accuracy > best_accuracy:
        print("model saved")
        best_accuracy = q3_accuracy
        # PATH = f"/home/ubuntu/instance1/datamodels/{batch_size}_{grad_accum}_{lr}_{epochs}_{round(q3_accuracy, 3)}_{round(t_loss, 3)}_cnn.pt"
        PATH = "model.pt"
        # torch.save({
        #             'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'loss': t_loss,
        #             }, PATH)
        torch.save(model.state_dict(), PATH)
      
      # freezes t5 language model
      if epoch == freeze_epoch:
        freeze_t5_model(model)


def train(model: torch.nn.Module,
          train_data: DataLoader,
          loss_fn,
          optimizer,
          grad_accum):
    """
    do a train on a minibatch
    """
    model.train()
    optimizer.zero_grad()
    total_loss = 0
    count = 0
    # batch accumulation parameter
    accum_iter = grad_accum
    for i, batch in enumerate(train_data):
        optimizer.zero_grad()
        # print(f"batch-{i}")
        ids, label, mask = batch
        ids = ids.to(device)
        mask = mask.to(device)
        # passes and weights update
        out = model(ids) # shape: [bs, max_seq_len, 3]
        # string to float conversion, padding and mask labels
        labels = process_labels(label, mask=mask, onehot=False).to(device)
        # reshape to make loss work 
        out = torch.transpose(out, 1, 2)
        assert out.shape[-1] == labels.shape[-1], f"out: {out.shape}, labels: {labels.shape}"
        loss = loss_fn(out, labels)
        loss.backward()
        total_loss += loss.item()
        count += 1
        wandb.log({"train_loss":loss.item()}) # logs loss for each batch

          # # weights update
          # if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_data)):
        optimizer.step()
    return total_loss/count

def validate(model: torch.nn.Module,
          val_data: DataLoader,
          loss_fn):
    model.eval()
    
    last_accuracy = 0
    total_loss = 0
    count = 0
    sum_accuracy = 0
    acc_list = []
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
      
      for batch_idx, out_logits in enumerate(out):
        # Calculate scores for each sequence individually
        # And average over them

        seqlen = len(label[batch_idx])
        preds = logits_to_preds(out_logits[:seqlen]) # already in form: [0, 1, 2, 3]
        true_label = label_to_id(label[batch_idx]) # convert label to machine readable.
        res_mask = mask[batch_idx][:seqlen] # [:seqlen] to cut the padding

        assert seqlen == len(preds) == len(res_mask), "length of seqs not matching"
        count += 1
        
        acc = q3_acc(true_label, preds, res_mask)
        sum_accuracy += acc
        acc_list.append(acc)
    # last_accuracy = sum_accuracy/len(val_data)# , np.std(acc_scores)
    # print("len acc scores: ", count, f"should be ({len(val_data)})")
    last_accuracy = sum(acc_list)/len(acc_list)
    return last_accuracy, total_loss/count, np.std(acc_list)

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
          print(f"mask:\t\t", mask)
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
                   seed: int, max_emb_size: int, tokenizer=None, masking=0) -> DataLoader:
    torch.manual_seed(seed)
    dataset = SequenceDataset(jsonl_path=jsonl_path,
                           device=device,
                           max_emb_size=max_emb_size,
                             tokenizer=tokenizer,
                             masking=masking)
    loader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=True, collate_fn=custom_collate)
    return loader

def freeze_t5_model(model):
    ## Freezes the t5 component of the t5 model
    print("freezing all but cnn")
    for layer, param in model.named_parameters():
        param.requires_grad = False
    # unfreeze CNN
    for layer, param in list(model.named_parameters())[-4:]:
        param.requires_grad = True

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using", device)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--grac", type=int, default=1)
    parser.add_argument("--maxemb", type=int, default=128)
    parser.add_argument("--optim", type=str, default="adamax")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--model_type", type=str, default="bert-cnn")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trainset", type=str, default="train.jsonl")
    parser.add_argument("--valset", type=str, default="val.jsonl")
    parser.add_argument("--wdnote", type=str)
    parser.add_argument("--trainable", type=int, default=24)
    parser.add_argument("--pn", type=str, default="runtesting")
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--fr", type=int, help="freezes t5 after epoch i", default=10)
    parser.add_argument("--msk", type=float, help="randomly mask the sequence", default=0)
    parser.add_argument("--datapath", type=str, help="path to datafolder", default="/notebooks/ss_pred_protrans_t5/data/")
    args = parser.parse_args()
    
    batch_size = args.bs
    grad_accum = args.grac
    max_emb_size = args.maxemb
    optimizer_name = args.optim
    lr = args.lr
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    epochs = args.epochs
    model_type = args.model_type
    dropout = args.dropout
    seed = args.seed
    trainset = args.trainset
    valset = args.valset
    wandb_note = args.wdnote
    trainable = args.trainable
    project_name = args.pn
    weight_decay = args.wd
    freeze_epoch = args.fr
    seq_mask = args.msk
    datapath = args.datapath


    # Choose model
    if model_type == "pt5-cnn":
      model = T5CNN().to(device)
      tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    elif model_type == "bert-cnn":
      model = BertCNN().to(device)
      tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")
    elif model_type == "pt5-lin":
      model = T5Linear(dropout=dropout).to(device)
      tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float16)
    elif model_type == "pbert-lin":
      model = BertLinear(dropout=dropout).to(device)
      tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")
    else:
      assert False, f"Model type not implemented {model_type}"
    
    ## Data loading
    train_path = datapath + trainset
    val_path = datapath + valset

    train_loader = get_dataloader(jsonl_path=train_path, 
                                  batch_size=batch_size, 
                                  device=device, seed=42,
                                  max_emb_size=max_emb_size, tokenizer=tokenizer,
                                 masking=seq_mask)

    val_loader = get_dataloader(jsonl_path=val_path, 
                                batch_size=1, 
                                device=device, seed=42,
                                max_emb_size=2000, tokenizer=tokenizer)

    

    # Apply freezing
    if args.trainable <= 0: ## -1 to freeze whole t5 model
        print("freeze all layers")
        freeze_t5_model(model)
    elif args.trainable == 24:
        print("No freezing supported for protbert")
    elif args.trainable > 0:
        num_trainable_layers = args.trainable
        trainable_t5_layers_stage1 = [str(integer) for integer in
                                      list(range(23, 23 - num_trainable_layers, -1))]
        print("Entering model freezing")
        for layer, param in model.named_parameters():
            param.requires_grad = False
        print("all layers frozen. Unfreezing trainable layers")
        unfr_c = 0

        # unfreeze desired layers
        for layer, param in model.named_parameters():
            lea = [trainable in layer for trainable in trainable_t5_layers_stage1]
            if sum(lea) >= 1:
                param.requires_grad = True
                unfr_c += 1
                # print(f"unfroze {layer}")
        # unfreeze CNN
        for layer, param in list(model.named_parameters())[-4:]:
            param.requires_grad = True
            unfr_c += 1
            # print(f"unfroze {layer}")

    # For testing and logging
    train_data = train_loader
    val_data = val_loader

    # wandb logging
    config = {"lr": str(lr).replace("0.", ""),
              "epochs": epochs,
              "batch_size": batch_size,
              "max_emb_size": max_emb_size,
              "grad_accum": grad_accum,
              "optim_name": optimizer_name,
              "model_type": model_type,
              "loss_fn": loss_fn,
              "dropout": dropout,
              "trainset": trainset,
              "valset": valset,
              "train_size": len(train_data),
              "val_size": len(val_data),
              "sequence_mask": seq_mask,
              "wandb_note": wandb_note,
              "number of trainable layers (freezing)": trainable,
              }
    experiment_name = f"{model_type}-{batch_size}_{lr}_{epochs}_{grad_accum}_{max_emb_size}_{wandb_note}"
    wandb.init(project=project_name, entity="kyttang", config=config, name=experiment_name)

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
                        loss_fn=loss_fn,
                        weight_decay=weight_decay,
                        freeze_epoch=freeze_epoch)
    
    ## Test data        
    # Test loader
    casp12_path = datapath + "casp12.jsonl"
    casp12_loader = get_dataloader(jsonl_path=casp12_path, batch_size=1, device=device, seed=seed,
                                 max_emb_size=5000, tokenizer=tokenizer)

    npis_path = datapath + "new_pisces.jsonl"
    npis_loader = get_dataloader(jsonl_path=npis_path, batch_size=1, device=device, seed=seed,
                                 max_emb_size=5000, tokenizer=tokenizer)
    
    
    ## Load model
    if model_type == "pt5-cnn":
      model = T5CNN().to(device)
    elif model_type == "bert-cnn":
      model = BertCNN(dropout=dropout).to(device)
    elif model_type == "pt5-lin":
      model = T5Linear(dropout=dropout).to(device)
    elif model_type == "pbert-lin":
      model = BertLinear(dropout=dropout).to(device)
    else:
      assert False, f"Model type not implemented {model_type}"
    
    model.load_state_dict(torch.load("model.pt"))
    model = model.to(device)
    print("new_pisces:", test(model, npis_loader, verbose=False))
    print("casp12:", test(model, casp12_loader, verbose=False))
    
    