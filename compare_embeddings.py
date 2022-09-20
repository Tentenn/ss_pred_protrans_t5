"""
Utilization:
python compare_embeddings.py --f1 <path_to_h5_file> --f2 <path_to_h5_file>

What it does:
compares these embeddings file using euclidean distance and cosine similarity
"""


import torch
from scipy import spatial
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

def cosine_sim_embedds(emb1, emb2):
  """
  Returns cosine similarity of two embeddings of shape Lx1024
  result = 0 means that they are the same
  The higher, the unsimilar the two vectors
  """
  emb1_rs = torch.reshape(emb1, (-1, ))
  emb2_rs = torch.reshape(emb2, (-1, ))
  return spatial.distance.cosine(emb1_rs, emb2_rs)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f1", type=str, help="Path to .h5 file")
    parser.add_argument("--f2", type=str, help="Path to .h5 file")
    args = parser.parse_args()
    
    embeddings1 = load_embeddings(args.f1)
    embeddings2 = load_embeddings(args.f2)

    cosim = []
    eudist = []
    
    for key in embeddings1.keys():
        emb1 = embeddings1[key]
        emb2 = embeddings2[key]
        cosim.append(cosine_sim_embedds(emb1, emb2))
        eudist.append(eucl_dist_embedds(emb1, emb2))
    
    cosim_df = pd.Series(cosim)
    eudist_df = pd.Series(eudist)
    
    print("Cosine Similarity:")
    print(cosim_df.describe())
    print("Euclidean Distance:")
    print(eudist_df.describe())
    
    n_samples = len(embeddings1.keys())
    
    plt.scatter([i for i in range(n_samples)], eudist, s=3)
    plt.savefig("eudist.png")
    plt.clf()
    plt.scatter([i for i in range(n_samples)], cosim, s=3)
    plt.savefig("cosim.png")