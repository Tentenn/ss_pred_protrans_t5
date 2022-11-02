"""
Generates embeddings given a t5 path 
Default t5 encoder half from Rostlab huggingface

Utilization: 
python gen_embeddings.py --m <prott5_model_path> --f <file> --p <parser_type>

"""


device = "cuda"

from torch.utils.data import DataLoader, Dataset
import torch
from transformers import T5Tokenizer, T5EncoderModel, BertModel, BertTokenizer
from pathlib import Path
from pyfaidx import Fasta
from typing import Dict, Tuple, List
import numpy as np
import re
from tqdm import tqdm
import gc
import time
import json
import h5py
import tqdm
import argparse

# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50) 
def get_T5_model(path):
    model = T5EncoderModel.from_pretrained(path)
    # model = BertModel.from_pretrained("Rostlab/prot_bert")
    # model = T5EncoderModel.from_pretrained("ss_pred_protrans_t5/pt5_lm_model.pt")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    # tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)

    return model, tokenizer

def get_Bert_model():
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    model = model.to(device) # move model to GPU
    model = model.eval() # set model to evaluation model
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
    return model, tokenizer

# Generate embeddings via batch-processing
# per_residue indicates that embeddings for each residue in a protein should be returned.
# per_protein indicates that embeddings for a whole protein should be returned (average-pooling)
# max_residues gives the upper limit of residues within one batch
# max_seq_len gives the upper sequences length for applying batch-processing
# max_batch gives the upper number of sequences per batch
def get_embeddings(model, tokenizer, seqs, max_residues=4000, 
                   max_seq_len=1000, max_batch=100):

    results = {"residue_embs" : dict(), 
               "protein_embs" : dict(),
               "sec_structs" : dict() 
               }

    # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
    seq_dict   = sorted( seqs.items(), key=lambda kv: len( seqs[kv[0]] ), reverse=True )
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict,1):
        print(seq_idx, "/", len(seq_dict), end= "\r")
        seq = seq
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id,seq,seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed 
        n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len 
        if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # add_special_tokens adds extra token at the end of each sequence
            token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
            
            try:
                with torch.no_grad():
                    # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue


            for batch_idx, identifier in enumerate(pdb_ids): # for each protein in the current mini-batch
                s_len = seq_lens[batch_idx]
                # slice off padding --> batch-size x seq_len x embedding_dim
                emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                results["residue_embs"][ identifier ] = emb.detach().cpu().numpy().squeeze()


    passed_time=time.time()-start
    avg_time = passed_time/(len(results["residue_embs"])+1)
    print('\n############# EMBEDDING STATS #############')
    print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
    print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
    print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
        passed_time/60, avg_time ))
    print('\n############# END #############')
    return results

def read_jsonl(jsonl_path):
    '''
        converts jsonl file to a dict: uniprot_id -> sequence 
    '''
    
    seqs = dict()
    with open(jsonl_path) as t:
     
      seq_dicts = [json.loads(line) for line in t]
      ## Converts the list of dicts (jsonl file) to a single dict with id -> sequence
      for d in seq_dicts:
        seqs[d["id"]] = d["sequence"]

    return seqs

def read_allseqs(seqs_path):
  '''
  Converts allseqs to dict
  '''
  seqs = dict()
  with open(seqs_path) as t:
    for line in t.read().split("\n"):
      try:
        id, seq = line.split("\t")
        seqs[id] = seq
      except ValueError:
        print("ERROR 01 Empty Line", line.split(" "))
  return seqs

def read_fasta(seqs_path):
  '''
  Converts allseqs to dict
  '''
  seqs = dict()
  with open(seqs_path) as t:
    for seg in t.read().split(">"):
      try:
        id, seq, _ = seg.split("\n")
        seqs[id] = seq.strip()
      except ValueError:
        print("ERROR 01 Empty Line", seg)
  return seqs

    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=str, default="Rostlab/prot_t5_xl_half_uniref50-enc", help="path to t5 model folder")
    parser.add_argument("--f", type=str, help="file: fasta,txt,jsonl")
    parser.add_argument("--p", type=str, help="parser mode options: fasta, jsonl, allseqs ")
    parser.add_argument("--g", type=str, help="model type: bert, t5")
    args = parser.parse_args()
                        
    # Load the encoder part of ProtT5-XL-U50 in half-precision (recommended)
    if args.g == "t5":
        print("generating embeddings with pt5")
        model, tokenizer = get_T5_model(args.m)
    elif args.g == "bert":
        print("generating embeddings with bert")
        model, tokenizer = get_Bert_model()
    else:
        assert False, f"Error 1, no model type {args.g} found"
    
    # data_dir_path = "drive/MyDrive/BachelorThesis/data/seth_disorder/"
    # data_dir_path = "./"
    data_paths = [args.f]# ["train.jsonl", "val.jsonl", "new_pisces.jsonl", "casp12.jsonl"]
    # data_paths = ["casp12.jsonl"]
    # Load fastas
    for data_path in data_paths:
      print(f"Creating embeddings for {data_path}")
      path = data_path # data_dir_path + data_path
      if args.p == "fasta":
        seqs = read_fasta(path)
      elif args.p == "jsonl":
        seqs = read_jsonl(path)
      elif args.p == "allseqs":
        seqs = read_allseqs(path)
        
      # Compute embeddings and/or secondary structure predictions
      results = get_embeddings(model, tokenizer, seqs)
      # write embeddings
      model_name = args.m.replace("/", "-")
      if args.g == "bert":
        model_name = "bert"
      out_path = data_path+f"-{model_name}-"+"_pt5.h5"
      with h5py.File(str(out_path), "w") as hf:
        for sequence_id, embedding in results["residue_embs"].items():
            # noinspection PyUnboundLocalVariable
            hf.create_dataset(sequence_id, data=embedding)

        