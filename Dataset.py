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
                 max_emb_size=200,
                tokenizer=None,
                masking=0,
                max_samples=-1):
        self.tokenizer = tokenizer
        self.masking = masking
        self.device = device
        self.max_emb_size = max_emb_size
        self.data_dict = utils.load_all_data(jsonl_path)
        
        # randomly throws out samples until % max_samples is reached
        if max_samples > 0: 
            throwout_mask = torch.rand((len(self.data_dict.keys()))) < max_samples
            for k,m in zip(list(self.data_dict.keys()), throwout_mask):
              if m.item():
                self.data_dict.pop(k)
            
        self.headers = tuple(self.data_dict.keys())
        # Header refers to uniprot id
        assert len(self.headers) == len(self.data_dict.keys()), "dict len not the same"

    def __getitem__(self, index: int): # -> Tuple[torch.Tensor, int, str]:
        # given an index, return ids, label, mask
        header = self.headers[index]
        sequence, label, mask = self.data_dict[header]
        # trim length to max_len
        if len(sequence) > self.max_emb_size:
          sequence, label, mask = self.retr_seg(sequence, label, mask, self.max_emb_size)
        
        
        prepro = str(re.sub(r"[UZOB]", "X", sequence))
        # sequence_sliced = " ".join(prepro)
        # print("TYPE OF SEQUENCE", type(prepro))
        # print(sequence)
        if self.masking>0:
            masked_seq = self.mask_sequence(seq=sequence, d=self.masking) # also adds spaces
        else:
            masked_seq = " ".join(prepro)
        # get ids
        ids = self.tokenizer.encode(masked_seq, add_special_tokens=True)
        # print(ids[:5], "...",ids[-5:], print(ids.size))
        
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
        return str(seq_s), label_s, mask_s
    
    def mask_sequence(self, seq, d=0.15):
        
      seq_s = [f" {c} " for c in seq]
      # print("#### seq_s", seq_s)
      seqlen = len(seq_s)
      ## create random masking indices based on d
      indices = sorted(random.sample(range(seqlen), k=int(d*seqlen)))
      ## replace sequence with '#' token
      newseq = "".join([" # " if i in indices else f" {seq_s[i]} " for i in range(seqlen)])
      ## replace '#' with ascending sentinel tokens
      iter = range(newseq.count("#"))
      for i, ind in enumerate(indices):
        seq_s[ind] = f" <extra_id_{iter[i]}> "
      ## join and remove double spaces
      return "".join(seq_s).replace("  ", " ").strip() 