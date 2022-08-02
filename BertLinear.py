"""
Adapted and modified from Michael Heinzinger Notebook
https://colab.research.google.com/drive/1TUj-ayG3WO52n5N50S7KH9vtt6zRkdmj?usp=sharing
"""

from transformers import BertModel
import torch

class BertLinear(torch.nn.Module):
    def __init__(self, dropout=0.25):
        super(BertLinear, self).__init__()

        self.bert = BertModel.from_pretrained("Rostlab/prot_bert")

        self.linear = torch.nn.Sequential(
            # torch.nn.Dropout(dropout),
            torch.nn.Linear(1024, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 3)
        )

    def forward(self, input_ids):
        # trim ids (last item of each seq)
        # input_ids = input_ids[:,:-1]
        # assert input_ids.shape == attention_mask.shape, f"input_ids: {input_ids.shape}, mask: {attention_mask.shape}"
        
        
        # create embeddings
        emb = self.bert(input_ids).last_hidden_state
        # trim last
        emb = emb[:, 1:-1, :]

        # old architecture
        # print("embsize", emb.size())
        # emb = emb.permute(0, 2, 1).unsqueeze(dim=-1)
        # print("embsize", emb.size())
        d3_Yhat = self.linear(emb)  # OUT: (B x 32 x L x 1)
        # d3_Yhat = self.dssp3_classifier(emb).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 3)
        # d3_Yhat = self.one_linear(emb)
        return d3_Yhat