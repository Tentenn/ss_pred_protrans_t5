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

def _filter_ids(ids_tensor):
    
    for ids in ids_tensor:
        i = 0
        for ind,e in enumerate(ids):
          # assert ind < len(ids), f"index out of range for {ind, len(ids)}"
          if e == 0:
            ids[ind] = tokenizer.additional_special_tokens_ids[0] ## TODO: Add more special tokens
            i += 1
    return ids_tensor

def apply_mask(input_ids, noise_mask, device):
  # Applies noise mask to ids
  # result ["M", "extra_id_0", "C", "C"] <= as ids
  assert len(np.copy(input_ids.cpu())) == len(input_ids), "copy length not the same"
  assert input_ids.shape == noise_mask.shape, f"input_ids: {input_ids} \n noise_mask: {noise_mask}"
  masked_input_ids = _filter_ids(copy.deepcopy(input_ids).to(device)*~noise_mask)
  # masked_labels_ids = _filter_ids(np.copy(input_ids)*noise_mask)
  # masked_labels_ids[-1] = 1
  # masked_input_ids.append(tokenizer.eos_token_id) # add eos token
  return torch.tensor(masked_input_ids), "not implemented"# masked_labels_ids

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
                       device):

    if optimizer_name == "adam":
      optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # elif optimizer_name == "adamax":
    #   optimizer = Adamax(model.parameters(), lr=lr)
    elif optimizer_name == "adafactor":
      optimizer = Adafactor([{"params":lm.parameters(), 'lr': 0.00001}, {"params":inf_model.parameters(), 'lr': 0.0001}], 
      lr=lr, relative_step=False, scale_parameter=False, weight_decay=weight_decay)
    # elif optimizer_name == "adafactor_rs":
    #   optimizer = Adafactor(model.parameters(), weight_decay=weight_decay)
    # elif optimizer_name == "adagrad":
    #   optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    # elif optimizer_name == "rmsprop":
    #   optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
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
      t_loss_sum, t_loss_lm, t_loss_inf = train(lm, inf_model, train_data, loss_fn, optimizer, grad_accum)
      print("t_loss_sum:",t_loss_sum,"t_loss_lm:",t_loss_sum,"t_loss_inf:",t_loss_lm)
      
      assert False, "[DEV] Testing"

      # validate results and calculate scores
      print(f"validate epoch {epoch}")
      q3_accuracy, v_loss = validate(model, val_data, loss_fn)
      wandb.log({"accuracy (Q3)":q3_accuracy})
      wandb.log({"val_loss":v_loss})
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


def train(lm: torch.nn.Module,
          inf_model: torch.nn.Module,
          train_data: DataLoader,
          loss_fn,
          optimizer,
          grad_accum):
    """
    do a train on a minibatch
    """
    model.train()
    optimizer.zero_grad()
    total_sum_loss = 0 # summed loss
    total_lm_loss = 0 # language model loss
    total_inf_loss = 0 # inference model loss
    count = 0
    # batch accumulation parameter
    accum_iter = grad_accum
    for i, batch in enumerate(train_data):
        optimizer.zero_grad()
        ids, label, mask = batch
        ids = ids.to(device)
        mask = mask.to(device)
        # string to float conversion, padding and mask labels
        labels = process_labels(label, mask=mask).to(device)

        # generate span masks and apply to ids
        noise_mask = torch.tensor([random_spans_noise_mask(len(single_ids), 0.1, 1) for single_ids in ids]).to(device)
        assert ids.shape == noise_mask.shape, "shape not the same length"
        masked_input, masked_labels = apply_mask(ids, noise_mask, device)
        masked_input = masked_input.to(device)
        
        # forward pass of whole language model
        assert len(ids.shape) == 2, "Shape not right"
        assert masked_input.shape == ids.shape, f"Shapes not match {masked_input.shape}, {ids.shape} \n {masked_input} \n {ids}"
        lm_output = lm(input_ids=masked_input, labels=torch.tensor(ids))
        lm_loss = lm_output.loss

        # get embeddings from lm output and pass through inference model
        embeddings = lm_output.encoder_last_hidden_state
        embeddings = remove_eos_embedding(embeddings)
        inf_out = inf_model(embeddings) # shape: [bs, max_seq_len, 3]
        # reshape to make loss work 
        inf_out = torch.transpose(inf_out, 1, 2)
        
        assert inf_out.shape[-1] == labels.shape[-1], f"out: {inf_out.shape}, labels: {labels.shape} \n {inf_out} \n {labels}"
        # assert inf_out.dtype == labels.dtype, f"not the same type {inf_out.dtype}, {labels.dtype}"
        inf_loss = loss_fn(inf_out, labels)
        
        # sum loss and do backward
        sum_loss = lm_loss + inf_loss
        sum_loss.backward()

        total_sum_loss += sum_loss.item()
        total_lm_loss += lm_loss.item()
        total_inf_loss += inf_loss.item()
        count += 1
        # print({"lm_loss":lm_loss.item(), 
        #           "inf_loss":inf_loss.item(), 
        #           "total_sum_loss":sum_loss.item()})
        wandb.log({"lm_loss":lm_loss.item(), 
                  "inf_loss":inf_loss.item(), 
                  "total_sum_loss":sum_loss.item()}) # logs loss for each batch

          # # weights update
          # if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_data)):
        optimizer.step()
    return total_sum_loss/count, total_lm_loss/count, total_inf_loss/count

def validate(lm: torch.nn.Module,
          inf_model: torch.nn.Module,
          val_data: DataLoader,
          loss_fn):
    model.eval()
    
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

      # generate span masks and apply to ids
      noise_mask = torch.tensor([random_spans_noise_mask(len(single_ids), 0.1, 1) for single_ids in ids]).to(device)
      assert ids.shape == noise_mask.shape, "shape not the same length"
      masked_input, masked_labels = apply_mask(ids, noise_mask, device)
      masked_input = masked_input.to(device)
      
      # forward pass of whole language model
      assert len(ids.shape) == 2, "Shape not right"
      assert masked_input.shape == ids.shape, f"Shapes not match {masked_input.shape}, {ids.shape} \n {masked_input} \n {ids}"
      lm_output = lm(input_ids=masked_input, labels=torch.tensor(ids))
      lm_loss = lm_output.loss
      total_lm_loss += lm_loss
      
      # get embeddings from lm output and pass through inference model
      embeddings = lm_output.encoder_last_hidden_state
      embeddings = remove_eos_embedding(embeddings)
      inf_out = inf_model(embeddings) # shape: [bs, max_seq_len, 3]
      # reshape to make loss work 
      inf_out = torch.transpose(inf_out, 1, 2)

      # calculate indeference loss
      inf_loss = loss_fn(inf_out, labels_f)
      total_inf_loss += inf_loss
      
      for batch_idx, out_logits in enumerate(inf_out):
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
    last_accuracy = sum_accuracy/len(val_data)# , np.std(acc_scores)
    # print("len acc scores: ", count, f"should be ({len(val_data)})")
    return last_accuracy, total_lm_loss/count, total_inf_loss

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
    parser.add_argument("--grac", type=int, default=10)
    parser.add_argument("--maxemb", type=int, default=400)
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
      tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
      model_cnn = ConvNet().to(device)
      model_pt5 = T5ForConditionalGeneration.from_pretrained("Rostlab/prot_t5_xl_uniref50")
      model_pt5 = model_pt5.to(device)
    # elif model_type == "pbert-cnn":
    #   model = ProtBertCNN(dropout=dropout).to(device)
    #   tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")
    # elif model_type == "pt5-lin":
    #   model = T5Linear(dropout=dropout).to(device)
    #   tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float16)
    # elif model_type == "pbert-lin":
    #   model = BertLinear(dropout=dropout).to(device)
    #   tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")
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

    # Test loader
    casp12_path = datapath + "casp12.jsonl"
    casp12_loader = get_dataloader(jsonl_path=casp12_path, batch_size=1, device=device, seed=seed,
                                 max_emb_size=5000, tokenizer=tokenizer)

    npis_path = datapath + "new_pisces.jsonl"
    npis_loader = get_dataloader(jsonl_path=npis_path, batch_size=1, device=device, seed=seed,
                                 max_emb_size=5000, tokenizer=tokenizer)

    # Apply freezing
    if args.trainable <= 0: ## -1 to freeze whole t5 model
        print("freeze all layers")
        freeze_t5_model(model)
    elif args.trainable > 0 and args.trainable < 24:
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
    
    os.environ["WANDB_API_KEY"] = "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"
    os.environ["WANDB_MODE"] = "online"
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
              "casp12_path": casp12_path,
              "npis_path": npis_path,
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
    main_training_loop(lm=model_pt5,
                        inf_model=model_cnn, 
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
    ## Load model
    if model_type == "pt5-cnn":
      model_cnn = ConvNet().to(device)
      model_pt5 = T5ForConditionalGeneration.to(device)
    # elif model_type == "pbert-cnn":
    #   model = ProtBertCNN(dropout=dropout).to(device)
    # elif model_type == "pt5-lin":
    #   model = T5Linear(dropout=dropout).to(device)
    # elif model_type == "pbert-lin":
    #   model = BertLinear(dropout=dropout).to(device)
    else:
      assert False, f"Model type not implemented {model_type}"
    
    model.load_state_dict(torch.load("model.pt"))
    model = model.to(device)
    print("new_pisces:", test(model, npis_loader, verbose=False))
    print("casp12:", test(model, casp12_loader, verbose=False))
    
    