"""
Adapted and modified from Michael Heinzinger Notebook
https://colab.research.google.com/drive/1TUj-ayG3WO52n5N50S7KH9vtt6zRkdmj?usp=sharing
"""

from transformers import T5EncoderModel, T5Tokenizer
import torch

class T5Linear(torch.nn.Module):
    def __init__(self, dropout=0.25):
        super(T5CNN, self).__init__()

        self.t5 = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")

        self.linear = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(1024, 512)
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 3)
        )

    def forward(self, input_ids):
        # trim ids (last item of each seq)
        input_ids = input_ids[:,:-1]
        # assert input_ids.shape == attention_mask.shape, f"input_ids: {input_ids.shape}, mask: {attention_mask.shape}"
        
        # create embeddings
        emb = self.t5(input_ids).last_hidden_state

        # old architecture
        emb = emb.permute(0, 2, 1).unsqueeze(dim=-1)
        d3_Yhat = self.linear(emb)  # OUT: (B x 32 x L x 1)
        # d3_Yhat = self.dssp3_classifier(emb).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 3)
        # d3_Yhat = self.one_linear(emb)
        return d3_Yhat