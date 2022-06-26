import utils
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple, List

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
        self.data_dict = utils.load_data(labels_path)
        self.headers = tuple(self.embeddings.keys())
        # Header refers to uniprot id
        assert len(self.headers) == len(self.data_dict.keys()), "dict len not the same"

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        # given an index, return embedding, label, mask
        header = self.headers[index]
        # assert header in self.embeddings.keys(), "key not in dict"
        embedding = self.embeddings[header]
        label, mask = self.data_dict[header]
        return embedding, label, mask

    def __len__(self) -> int:
        return len(self.headers)