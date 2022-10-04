"""
Utilization:
python compare_embeddings.py --f1 <path_to_h5_file> --f2 <path_to_h5_file> --f3 <path_to_h5_file>

What it does:
compares these embeddings file using euclidean distance and cosine similarity.
The first on is the embeddings it is compared to
"""


import torch
from scipy import spatial
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import argparse

def cosine_sim_embedds(emb1, emb2):
  """
  Returns cosine similarity of two embeddings of shape Lx1024
  result = 0 means that they are the same
  The higher, the unsimilar the two vectors
  """
  emb1_rs = torch.reshape(emb1, (-1, ))
  emb2_rs = torch.reshape(emb2, (-1, ))
  return 1 - spatial.distance.cosine(emb1_rs, emb2_rs)

def eucl_dist_embedds(emb1, emb2):
  """
  Returns euclidean distance of two embeddings of shape Lx1024
  result = 0 means that they are the same, 
  The higher the unsimilar the two vectors
  """
  emb1_rs = torch.reshape(emb1, (-1, ))
  emb2_rs = torch.reshape(emb2, (-1, ))
  pdist = torch.nn.PairwiseDistance(p=2)
  return pdist(emb1_rs, emb2_rs).item()

def load_embeddings(path):
  with h5py.File(path, 'r') as f:
    embeddings_dict = {seq_identifier: torch.tensor(np.array(f[seq_identifier])) for seq_identifier in f.keys()}
  return embeddings_dict

def describe_difference(embeddings1, embeddings2, name="", verbose=False):
    cosim = []
    eudist = []
    for key in embeddings1.keys():
        emb1 = embeddings1[key]
        emb2 = embeddings2[key]
        cosim.append(cosine_sim_embedds(emb1, emb2))
        eudist.append(eucl_dist_embedds(emb1, emb2))
    
    cosim_df = pd.Series(cosim)
    eudist_df = pd.Series(eudist)
    
    if verbose:
        print(f"Cosine Similarity between {name}:")
        print(cosim_df.describe())
        print(f"Euclidean Distance between {name}:")
        print(eudist_df.describe())
    
    return cosim, eudist
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f1", type=str, help="Path to .h5 file, f2 and f3 are compared to f1")
    parser.add_argument("--f2", type=str, help="Path to .h5 file")
    parser.add_argument("--f3", type=str, help="Path to .h5 file")
    args = parser.parse_args()
    
    embeddings1 = load_embeddings(args.f1)
    embeddings2 = load_embeddings(args.f2)
    embeddings3 = load_embeddings(args.f3)

    cosim12, eudist12 = describe_difference(embeddings1, embeddings2, name="lr=5e-05")
    cosim13, eudist13 = describe_difference(embeddings1, embeddings3, name="lr=0.001")
    
    
    n_samples = len(embeddings1.keys())
    # x_axis = ["pt5-ft_lr=5e-05" for i in range(n_samples)] + ["pt5-ft_lr=0.001" for i in range(n_samples)]
    x_axis = [i for i in range(n_samples)] + [i for i in range(n_samples)]
    colors = ["blue" for _ in range(n_samples)] + ["red" for _ in range(n_samples)]
    eudist_y_axis = eudist12 + eudist13
    cosim_y_axis = cosim12 + cosim13
    
    plt.scatter(x_axis, eudist_y_axis, s=3, color=colors)
    plt.savefig("eudist.png")
    plt.clf()
    plt.scatter(x_axis, cosim_y_axis, s=3, color=colors)
    plt.savefig("cosim.png")