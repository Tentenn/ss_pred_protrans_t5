import json
import torch
import h5py
import numpy as np

def load_data(jsonl_path):
  """
  :param jsonl_path: path to jsonl containing info about id, sequence, labels and mask
  :return: dict containing all
  example: 5t87-E CCCCCCHHHHHHCCEEEECCEEEEEEECCCCCCEEEEEEEECCEEEEEEECHHHCCHHHHHHHHCCCCCCCCCCCC
  """
  with open(jsonl_path) as t:
    data_dict = dict()
    ## Converts the list of dicts (jsonl file) to a single dict with id -> sequence
    for d in [json.loads(line) for line in t]:
      data_dict[d["id"]] = d["label"], d["resolved"]
  return data_dict

def load_all_data(jsonl_path):
  """
  :param jsonl_path: path to jsonl containing info about id, sequence, labels and mask
  :return: dict containing sequence, label and resolved mask
  """
  with open(jsonl_path) as t:
    data_dict = dict()
    ## Converts the list of dicts (jsonl file) to a single dict with id -> sequence
    for d in [json.loads(line) for line in t]:
      data_dict[d["id"]] = d["sequence"], d["label"], d["resolved"]
  return data_dict


def load_embeddings(embeddings_path):
  with h5py.File(embeddings_path, 'r') as f:
    embeddings_dict = {seq_identifier: torch.tensor(np.array(f[seq_identifier])) for seq_identifier in f.keys()}
  return embeddings_dict

def add_noise_embedding(embedding, device, density=0.2, mode="dropout", std=0.1, variance=0.5):
  if mode=="dropout":
    mask = torch.rand(embedding.shape, device=device) < 1 - density
    return embedding * mask
  elif mode=="noise":
    return embedding + (std**variance)*torch.randn(embedding.shape, device=device)
  elif mode=="residue":
    mask = torch.randn(embedding.shape[:2]).ge(density).unsqueeze(-1)
    return mask*embedding
  else:
    assert False, "Not implemented dropout mode '{mode}'. available dropout, noise, residue"