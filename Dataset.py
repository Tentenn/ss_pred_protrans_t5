from torch._C import StringType
import utils
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple, List
from transformers import T5Tokenizer, BertTokenizer
import re
import random

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

class SequenceDataset(Dataset):
    """
    SequenceDataset:
    This dataset returns the sequence, labels and mask
    """

    def __init__(self, jsonl_path: str,
                 device: torch.device,
                 max_emb_size=200):
        self.device = device
        self.max_emb_size = max_emb_size
        self.data_dict = utils.load_all_data(jsonl_path)
        self.headers = tuple(self.data_dict.keys())
        # Header refers to uniprot id
        assert len(self.headers) == len(self.data_dict.keys()), "dict len not the same"

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        # given an index, return ids, label, mask
        header = self.headers[index]
        sequence, label, mask = self.data_dict[header]
        # tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")

        
        # print(f"{type(label)}, {type(mask)}")
        # Preprocess sequence to ProtTrans format
        if len(sequence) > self.max_emb_size:
          sequence, label, mask = self.retr_seg(sequence, label, mask, self.max_emb_size)

        sequence = " ".join(sequence)
        prepro = re.sub(r"[UZOB]", "X", sequence)
        ids = tokenizer.encode(prepro)
        
        assert len(label) == len(mask), "label and mask Not the same length (__getitem__)"
        return ids, label, mask

    def __len__(self) -> int:
        return len(self.headers)
    
    def retr_seg(self, seq: str, label:str, mask:str, max_len: int, mode="default"):
      if mode == "default":
        start_idx = random.randrange(0, len(seq)-max_len)
        # print(start_idx)
        seq_s = seq[start_idx:start_idx+max_len]
        label_s = label[start_idx:start_idx+max_len]
        mask_s = mask[start_idx:start_idx+max_len]
        return seq_s, label_s, mask_s