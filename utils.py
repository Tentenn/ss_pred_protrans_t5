import json
import torch
import h5py
import numpy as np

def load_labels(jsonl_path):
    """
    :param jsonl_path: path to jsonl containing info about id, sequence, labels and mask
    :return: dict containing all
    example: 5t87-E CCCCCCHHHHHHCCEEEECCEEEEEEECCCCCCEEEEEEEECCEEEEEEECHHHCCHHHHHHHHCCCCCCCCCCCC
    """
    with open(jsonl_path) as t:
        seq_dicts = [json.loads(line) for line in t]
        labels_dict = dict()
        ## Converts the list of dicts (jsonl file) to a single dict with id -> sequence

        for d in seq_dicts:
            labels_dict[d["id"]] = d["label"]
    return labels_dict

def load_embeddings(embeddings_path):
    with h5py.File(embeddings_path, 'r') as f:
        embeddings_dict = {seq_identifier: torch.tensor(np.array(f[seq_identifier])) for seq_identifier in f.keys()}
    return embeddings_dict

if __name__ == "__main__":
    lb_dict = load_labels("data/casp12.jsonl")
    for k,v in lb_dict.items():
        print(k,v)
