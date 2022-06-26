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

def load_embeddings(embeddings_path):
  with h5py.File(embeddings_path, 'r') as f:
    embeddings_dict = {seq_identifier: torch.tensor(np.array(f[seq_identifier])) for seq_identifier in f.keys()}
  return embeddings_dict

def load_embeddings2(embeddings_path):
  embeddings_dict = dict()
  with h5py.File(embeddings_path, 'r') as f:
    for k in f.keys():
      embeddings_dict[k] = f[k]
  return embeddings_dict