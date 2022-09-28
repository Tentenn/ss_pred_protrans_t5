"""
This code trains the CNN for 3-State secondary structure prediction
Using ProtTrans T5 per residue embeddings.

For multi objective training, the protein language model 
is fine-tuned using: masked language modeling, fine-tuning with 
a simple CNN and a jsonl dataset.


"""

import torch
from torch import nn
import torch.optim as optim
from typing import Any
import json
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import T5EncoderModel, T5Tokenizer, BertTokenizer, T5ForConditionalGeneration
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
import copy
import os

import utils
from Dataset import SequenceDataset
from T5ConvNet import T5CNN
from T5Linear import T5Linear
from ConvNet import ConvNet
from smart_optim import Adamax
from transformers import Adafactor
from BertLinear import BertLinear


CLASS_MAPPING = {"H":0, "E":1, "L":2, "C":2}

def process_labels(labels: list, mask:list, onehot=False):
  """
  turns a list of labels labels and adds padding
  labels: example: ['HEECCC', 'HHHEEEECCC']
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

def _filter_ids(ids_tensor):
    
    for ids in ids_tensor:
        i = 0
        for ind,e in enumerate(ids):
          # assert ind < len(ids), f"index out of range for {ind, len(ids)}"
          if e == 0:
            if i < len(tokenizer.additional_special_tokens_ids):
                i = 0
            ids[ind] = tokenizer.additional_special_tokens_ids[i] ## TODO: Add more special tokens
            i += 1
    return ids_tensor

def apply_mask(input_ids, noise_mask, device):
  # Applies noise mask to ids
  # result ["M", "extra_id_0", "C", "C"] <= as ids
  assert len(np.copy(input_ids.cpu())) == len(input_ids), "copy length not the same"
  assert input_ids.shape == noise_mask.shape, f"input_ids: {input_ids} \n noise_mask: {noise_mask}"
  masked_input_ids = _filter_ids(copy.deepcopy(input_ids).to(device)*~noise_mask)
  return masked_input_ids, "not implemented"# masked_labels_ids

def logits_to_preds(logits):
  """
  @param logits: a tensor of size (seqlen, 3) containing logits of ss preds
  @returns: a list of predictions eg. [2, 1, 1, 2, 2, 2, 0] 
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

def random_spans_noise_mask(length, noise_density, mean_noise_span_length):

        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

def remove_eos_embedding(embedding):
  last_index = embedding.shape[1] - 1
  return embedding[:,:last_index]

def main_training_loop(lm: torch.nn.Module, # Language model
                       inf_model: torch.nn.Module, # inferece model for downstream
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
                       device,
                       inf_lr: float,
                       lm_lr: float,
                       valstep: int,
                       valsize: int,
                       val_path: str,
                       emb_d: float,
                       emb_d_mode: str,
                       lm_chkpt: str,
                       inf_chkpt: str):

    if optimizer_name == "adam":
      optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "adafactor":
      optimizer = Adafactor([{"params":lm.parameters(), 'lr': 0.00001}, {"params":inf_model.parameters(), 'lr': 0.0001}], 
      lr=lr, relative_step=False, scale_parameter=False, weight_decay=weight_decay)
    elif optimizer_name == "mixed":
      optimizer_lm = Adafactor(lm.parameters(), lr=lm_lr, relative_step=False, scale_parameter=False, weight_decay=weight_decay)
      optimizer_inf = torch.optim.Adam(inf_model.parameters(), lr=inf_lr, amsgrad=True)
      t_total = len(train_data) * epochs
      optimizer = None
    else:
      assert False, f"Optimizer {optimizer_name} not implemented"
    
    # track best scores
    best_accuracy = float('-inf')
    epochs_without_improvement = 0
    best_vloss = 10000
    best_accuracy = 0
    
    for epoch in range(epochs):
      t1_time = datetime.now()
      # train model and save train loss
      print(f"train epoch {epoch}")
      if optimizer_name == "mixed": # mixed = dual optimizer
        t_loss_sum, t_loss_lm, t_loss_inf = train(lm, inf_model, train_data, loss_fn, optimizer, 
                                                  grad_accum, optimizer_lm=optimizer_lm,
                                                  optimizer_inf=optimizer_inf, mixed=True,
                                                  valstep=valstep, valsize=valsize,
                                                  val_path=val_path, emb_d=emb_d,
                                                  emb_d_mode=emb_d_mode, val_data=val_data)
      else:
        print("using single optimizer for lm and inf model")
        t_loss_sum, t_loss_lm, t_loss_inf = train(lm, inf_model, 
                                                  train_data, loss_fn, optimizer, grad_accum,
                                                  valstep=valstep, valsize=valsize,
                                                  val_path=val_path, emb_d=emb_d,
                                                  emb_d_mode=emb_d_mode, val_data=val_data)
      print("t_loss_sum:",t_loss_sum,"t_loss_lm:",t_loss_lm,"t_loss_inf:",t_loss_inf)

      # validate results and calculate scores
      print(f"validate epoch {epoch}")
      q3_accuracy, v_loss_lm, v_loss_inf = validate(lm, inf_model, val_data, loss_fn)
      wandb.log({"accuracy (Q3)":q3_accuracy})
      wandb.log({"val_loss_lm":v_loss_lm})
      wandb.log({"val_loss_inf":v_loss_inf})
      now = datetime.now()
      print("acc:", q3_accuracy, "val_loss_lm:", v_loss_lm, "val_loss_inf:", v_loss_inf, "compute_time:", now-t1_time)
    
      # update best vloss
      if v_loss_inf < best_vloss and q3_accuracy > best_accuracy:
        print(f"new best models found with {round(v_loss_inf, 3)} and {round(q3_accuracy, 3)}. Saving models...")
        best_accuracy = q3_accuracy
        best_vloss = v_loss_inf
        epochs_without_improvement = 0
        best_accuracy = q3_accuracy
        lm.save_pretrained(lm_chkpt)
        torch.save(inf_model.state_dict(), inf_chkpt)
        print(f"models successfully saved as {lm_chkpt} and {inf_chkpt} at epoch {epoch}")
      else:
        epochs_without_improvement += 1
        print(f"Epochs without improvement: {epochs_without_improvement}")
        if epochs_without_improvement >= 3:
            print("max amount of epochs (3) without improvement reached. Stopping training...")
            break
        
      
      # freezes t5 language model
      if epoch == freeze_epoch:
        freeze_t5_model(lm)
      t2_time = datetime.now()
      print("Total time for this epoch:", t2_time-t1_time)


def train(lm: torch.nn.Module,
          inf_model: torch.nn.Module,
          train_data: DataLoader,
          loss_fn,
          optimizer,
          grad_accum,
          optimizer_lm=None,
          optimizer_inf=None,
          mixed=False,
          valstep=100,
          valsize=-1,
          val_path=None, 
          emb_d=0.2, 
          emb_d_mode="default",
          val_data=None):
    """
    do a train on a minibatch
    mixed: whether or not 2 optimizers are used
    """
    inf_model.train()
    lm.train()
    
    total_sum_loss = 0 # summed loss
    total_lm_loss = 0 # language model loss
    total_inf_loss = 0 # inference model loss
    count = 0
    # batch accumulation parameter
    accum_iter = grad_accum
    
    for i, batch in enumerate(train_data):
        # Perform mid-train validation step. Can be skipped to speed up training
        if i%valstep==0:
            if valsize==-1:
                mid_val_loader = val_data
            elif 0<valsize<1:
                mid_val_loader = get_dataloader(jsonl_path=val_path, 
                                batch_size=1, 
                                device=device, seed=42,
                                max_emb_size=2000, 
                                tokenizer=tokenizer,
                                max_samples=valsize)
            else:
                assert False, f"val_size must be between 0 and 1 or -1 was given {valsize}"
            t1 = datetime.now()
            mid_q3_accuracy, mid_v_loss_lm, mid_v_loss_inf = validate(lm, inf_model, mid_val_loader, loss_fn)
            wandb.log({"mid_q3_accuracy":mid_q3_accuracy, "mid_v_loss_inf":mid_v_loss_inf})
            gc.collect()
            inf_model.train()
            lm.train()
            now = datetime.now()
            print(f"[{now.strftime("%H:%M:%S")}] Step {i} of {len(train_data)} n: {len(mid_val_loader)} acc: {round(mid_q3_accuracy, 3)} vloss: {round(mid_v_loss_inf, 3)} time: {now-t1}")
            
        
        # Check if using dual optimizer
        if mixed:
            optimizer_lm.zero_grad()
            optimizer_inf.zero_grad()
        else:
            optimizer.zero_grad()
            
        ids, label, mask = batch
        ids = ids.to(device)
        mask = mask.to(device)
        # string to float conversion, padding and mask labels
        labels = process_labels(label, mask=mask).to(device)

        # generate span masks and apply to ids
        noise_mask = torch.tensor(np.array([random_spans_noise_mask(len(single_ids), 0.15, 1) for single_ids in ids])).to(device)
        assert ids.shape == noise_mask.shape, "shape not the same length"
        masked_input, masked_labels = apply_mask(ids, noise_mask, device)
        masked_input = masked_input.to(device)
        
        # LM LOSS: forward pass of whole language model
        assert len(ids.shape) == 2, "Shape not right"
        assert masked_input.shape == ids.shape, f"Shapes not match {masked_input.shape}, {ids.shape} \n {masked_input} \n {ids}"
        lm_output = lm(input_ids=masked_input, labels=ids)
        lm_loss = lm_output.loss

        # get embeddings from lm output and pass through inference model
        embeddings = lm_output.encoder_last_hidden_state
        # postprocessing embeddings
        embeddings = remove_eos_embedding(embeddings)
        embeddings = utils.add_noise_embedding(embeddings, device=device, density=emb_d, mode=emb_d_mode)
        inf_out = inf_model(embeddings) # shape: [bs, max_seq_len, 3]
        
        # INF LOSS: reshape to make loss work 
        inf_out = torch.transpose(inf_out, 1, 2)
        assert inf_out.shape[-1] == labels.shape[-1], f"out: {inf_out.shape}, labels: {labels.shape} \n {inf_out} \n {labels}"
        inf_loss = loss_fn(inf_out, labels)
        
        sum_loss = lm_loss + inf_loss
        
        # backward() + step()
        if mixed: # case: dual optimizer
            optimizer_lm.zero_grad()
            optimizer_inf.zero_grad()
            sum_loss.backward()
            optimizer_lm.step()
            optimizer_inf.step()
        else: # case: single optimizer
            assert optimizer != None, "No optimizer found durin backward"
            sum_loss.backward()
            optimizer.step()

        total_sum_loss += sum_loss.item()
        total_lm_loss += lm_loss.item()
        total_inf_loss += inf_loss.item()
        count += 1
        wandb.log({"lm_loss":lm_loss.item(), 
                  "inf_loss":inf_loss.item(), 
                  "total_sum_loss":sum_loss.item()}) # logs loss for each batch

          # # weights update
          # if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_data)):
        
        ## log optimizer learning rate
        # wandb.log({"inf_lr": optimizer_inf.get_lr()[0]})
        
    return total_sum_loss/count, total_lm_loss/count, total_inf_loss/count

def validate(lm: torch.nn.Module,
          inf_model: torch.nn.Module,
          val_data: DataLoader,
          loss_fn):
    lm.eval()
    inf_model.eval()
    
    last_accuracy = 0
    total_inf_loss = 0
    total_lm_loss = 0
    count = 0
    sum_accuracy = 0
    for i, batch in enumerate(val_data):
      ids, label, mask = batch

      labels_f = process_labels(label, mask=mask, onehot=False).to(device)
      ids = ids.to(device)
      mask = mask.to(device)

      # # generate span masks and apply to ids
      # noise_mask = torch.tensor(np.array([random_spans_noise_mask(len(single_ids), 0.1, 1) for single_ids in ids])).to(device)
      # assert ids.shape == noise_mask.shape, "shape not the same length"
      # masked_input, masked_labels = apply_mask(ids, noise_mask, device)
      # masked_input = masked_input.to(device)
      
      # forward pass of whole language model
      assert len(ids.shape) == 2, "Shape not right"
      with torch.no_grad():
        lm_output = lm(input_ids=ids, labels=ids)
      lm_loss = lm_output.loss
      total_lm_loss += lm_loss.item() # loss not needed here
      
      # get embeddings from lm output and pass through inference model
      embeddings = lm_output.encoder_last_hidden_state
      embeddings = remove_eos_embedding(embeddings)
      with torch.no_grad():
        inf_out = inf_model(embeddings) # shape: [bs, max_seq_len, 3]
    
      # reshape to make loss work 
      inf_out_for_loss = torch.transpose(inf_out, 1, 2)

      # calculate indeference loss
      inf_loss = loss_fn(inf_out_for_loss, labels_f)
      total_inf_loss += inf_loss.item()
      
      for batch_idx, out_logits in enumerate(inf_out):
        # Calculate scores for each sequence individually
        # And average over them

        seqlen = len(label[batch_idx])
        preds = logits_to_preds(out_logits[:seqlen]) # already in form: [0, 1, 2, 3]
        true_label = label_to_id(label[batch_idx]) # convert label to machine readable.
        res_mask = mask[batch_idx][:seqlen] # [:seqlen] to cut the padding

        assert seqlen == len(preds) == len(res_mask), f"length of seqs not matching: {len(preds)} and {len(res_mask)}"
        count += 1
        
        acc = q3_acc(true_label, preds, res_mask)
        sum_accuracy += acc
    last_accuracy = sum_accuracy/len(val_data)# , np.std(acc_scores)
    return last_accuracy, total_lm_loss/count, total_inf_loss/count

def test(lm: torch.nn.Module,
         inf_model: torch.nn.Module,
         test_data: DataLoader,
         verbose=False):
    """
    verbose argument: whether or not to show actual predictions
    """
    lm.eval()
    inf_model.eval()
    
    acc_scores = []
    last_accuracy = 0
    total_inf_loss = 0
    total_lm_loss = 0
    count = 0
    sum_accuracy = 0
    for i, batch in enumerate(test_data):
      ids, label, mask = batch

      labels_f = process_labels(label, mask=mask, onehot=False).to(device)
      ids = ids.to(device)
      mask = mask.to(device)
    
      # get lm output 
      with torch.no_grad():
        lm_output = lm(input_ids=ids, labels=ids)
      
      # get embeddings from lm output and pass through inference model
      embeddings = lm_output.encoder_last_hidden_state
      embeddings = remove_eos_embedding(embeddings)
      with torch.no_grad():
        inf_out = inf_model(embeddings) # shape: [bs, max_seq_len, 3]
    
      # reshape to make loss work 
      inf_out_for_loss = torch.transpose(inf_out, 1, 2)

      # calculate indeference loss
      inf_loss = loss_fn(inf_out_for_loss, labels_f)
      total_inf_loss += inf_loss.item()
        
      for batch_idx, out_logits in enumerate(inf_out):
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
      inputs = pad_sequence(inputs, batch_first=True) # pad to longest batch
      
      labels = [d[1] for d in data]
      res_mask = [torch.tensor(np.array([float(dig) for dig in d[2]])) for d in data]
      mask = pad_sequence(res_mask, batch_first=True)
      
      
      return inputs, labels, mask


def get_dataloader(jsonl_path: str, batch_size: int, device: torch.device,
                   seed: int, max_emb_size: int, tokenizer=None, masking=0, max_samples=-1) -> DataLoader:
    torch.manual_seed(seed)
    dataset = SequenceDataset(jsonl_path=jsonl_path,
                           device=device,
                           max_emb_size=max_emb_size,
                             tokenizer=tokenizer,
                             masking=masking, max_samples=max_samples)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--grac", type=int, default=1)
    parser.add_argument("--maxemb", type=int, default=128)
    parser.add_argument("--optim", type=str, default="adamax")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--model_type", type=str, default="pt5-cnn")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trainset", type=str, default="new_pisces_200.jsonl")
    parser.add_argument("--valset", type=str, default="casp12_200.jsonl")
    parser.add_argument("--wdnote", type=str)
    parser.add_argument("--trainable", type=int, default=24)
    parser.add_argument("--pn", type=str, default="runtesting")
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--fr", type=int, help="freezes t5 after epoch i", default=10)
    parser.add_argument("--msk", type=float, help="randomly mask the sequence", default=0)
    parser.add_argument("--datapath", type=str, help="path to datafolder", default="/home/ubuntu/instance1/data/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument('--run_test', default=True, action=argparse.BooleanOptionalAction) ## Not Working!
    parser.add_argument("--lm_lr", type=float, default=0.0001)
    parser.add_argument("--inf_lr", type=float, default=0.0001)
    parser.add_argument("--valstep", type=int, default=50, help="do a validation after n steps")
    parser.add_argument("--valsize", type=float, default=-1, help="size of mini validation, -1 for all val data, 0<x<1 for fraction")
    parser.add_argument("--emb_d", type=float, default=0, help="variable for density of embedding dropout")
    parser.add_argument("--emb_d_mode", type=str, default="dropout", help="available modes: noise, dropout, residue")
    parser.add_argument("--lm_chkpt", type=str, default="pt5_lm_model.pt", help="name of checkpoint file of language model (.pt)")
    parser.add_argument("--inf_chkpt", type=str, default="cnn_inf_model.pt", help="name of checkpoint file of inference model (.pt)")
    parser.add_argument("--from_chkpt_lm", type=str, default=None, help="start training from a language model checkpoint")
    parser.add_argument("--from_chkpt_inf", type=str, default=None, help="start training from a inference checkpoint")
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
    device_name = args.device
    run_test = args.run_test
    lm_lr = args.lm_lr
    inf_lr = args.inf_lr
    valstep = args.valstep
    valsize = args.valsize
    emb_d = args.emb_d
    emb_d_mode = args.emb_d_mode
    lm_chkpt = args.lm_chkpt
    inf_chkpt = args.inf_chkpt
    
    ## Chose device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device_name == "cpu":
        device = torch.device("cpu")
    print("Using", device)

    ## Choose model
    print("load model...")
    if model_type == "pt5-cnn":
      tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
      model_cnn = ConvNet()
      if args.from_chkpt_inf is not None:
        print(f"Starting training of inference model from a checkpoint {args.from_chkpt_inf}")
        model_cnn.load_state_dict(torch.load(args.from_chkpt_inf))
      model_cnn = model_cnn.to(device)
        
      if args.from_chkpt_lm is not None:
        print(f"Starting training of language model from a checkpoint {args.from_chkpt_lm}")
        model_pt5 = T5ForConditionalGeneration.from_pretrained(args.from_chkpt_lm).to(device)
      else:
        model_pt5 = T5ForConditionalGeneration.from_pretrained("Rostlab/prot_t5_xl_uniref50").to(device)
    else:
      assert False, f"Model type not implemented {model_type}"
    
    ## Data loading
    print("load data...")
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

    ## Apply freezing
    if args.trainable <= 0: ## -1 to freeze whole t5 model
        print("freeze all layers")
        freeze_t5_model(model)
    elif args.trainable > 0 and args.trainable < 24:
        num_trainable_layers = args.trainable
        trainable_t5_layers_stage1 = [str(integer) for integer in
                                      list(range(23, 23 - num_trainable_layers, -1))]
        print("Entering model freezing")
        for layer, param in model_pt5.named_parameters():
            param.requires_grad = False
        print("all layers frozen. Unfreezing trainable layers")
        unfr_c = 0

        # unfreeze desired layers
        for layer, param in model_pt5.named_parameters():
            lea = [trainable in layer for trainable in trainable_t5_layers_stage1]
            if sum(lea) >= 1:
                param.requires_grad = True
                unfr_c += 1
                # print(f"unfroze {layer}")
    else:
        print("No freezing")

    ## wandb logging
    config = {"lr": str(lr).replace("0.", ""),
              "inf_lr": inf_lr,
              "lm_lr": lm_lr,
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
              "train_size": len(train_loader),
              "val_size": len(val_loader),
              "sequence_mask": seq_mask,
              "wandb_note": wandb_note,
              "number of trainable layers (freezing)": trainable,
              }
    experiment_name = f"{wandb_note}_{model_type}-{batch_size}_{epochs}_{max_emb_size}_{lm_lr}_{inf_lr}_{random.randint(300, 999)}"
    wandb.init(project=project_name, entity="kyttang", config=config, name=experiment_name)

    ## start training
    print("start training...")
    main_training_loop(lm=model_pt5,
                        inf_model=model_cnn, 
                        train_data=train_loader, 
                        val_data=val_loader, 
                        device=device, 
                        batch_size=batch_size,
                        lr=lr,
                        epochs=epochs,
                        grad_accum=grad_accum,
                        optimizer_name=optimizer_name,
                        loss_fn=loss_fn,
                        weight_decay=weight_decay,
                        freeze_epoch=freeze_epoch,
                        inf_lr=inf_lr,
                        lm_lr=lm_lr,
                        valstep=valstep,
                        valsize=valsize,
                        val_path=val_path,
                        emb_d=emb_d,
                        emb_d_mode=emb_d_mode,
                        lm_chkpt=lm_chkpt,
                        inf_chkpt=inf_chkpt)
    
    ## Test data        
    if run_test:
        print("start testing...")
        # Test loader
        casp12_path = datapath + "casp12.jsonl"
        casp12_loader = get_dataloader(jsonl_path=casp12_path, batch_size=1, 
                                       device=device, seed=seed,
                                     max_emb_size=5000, tokenizer=tokenizer)
        npis_path = datapath + "new_pisces.jsonl"
        npis_loader = get_dataloader(jsonl_path=npis_path, batch_size=1, 
                                     device=device, seed=seed,
                                     max_emb_size=5000, tokenizer=tokenizer)
        ## Load model
        if model_type == "pt5-cnn":
          model_inf = ConvNet()
          model_inf.load_state_dict(torch.load(inf_chkpt))
          model_pt5 = T5ForConditionalGeneration.from_pretrained(lm_chkpt)
        else:
          assert False, f"Model type not implemented {model_type}"

        model_pt5 = model_pt5.to(device)
        model_inf = model_inf.to(device)
        print("new_pisces:", test(model_pt5, model_inf, npis_loader, verbose=False))
        print("casp12:", test(model_pt5, model_inf, casp12_loader, verbose=False))
    
    