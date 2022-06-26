"""
Adapted and modified from Michael Heinzinger Notebook
https://colab.research.google.com/drive/1TUj-ayG3WO52n5N50S7KH9vtt6zRkdmj?usp=sharing
"""

from transformers import T5EncoderModel, T5Tokenizer
import torch

class T5CNN(torch.nn.Module):
    def __init__(self):
        super(T5CNN, self).__init__()

        self.t5 = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")

        self.elmo_feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
        )
        n_final_in = 32
        self.dssp3_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 3, kernel_size=(7, 1), padding=(3, 0))  # 7
        )

    def forward(self, input_ids, attention_mask):
        # trim ids (last item of each seq)
        input_ids = input_ids[:,:-1]
        assert input_ids.shape == attention_mask.shape, f"input_ids: {input_ids.shape}, mask: {attention_mask.shape}"
        
        # create embeddings
        emb = self.t5(input_ids, attention_mask=attention_mask).last_hidden_state

        # remove special tokens

        # dropout?

        # padding

        # old architecture
        emb = emb.permute(0, 2, 1).unsqueeze(dim=-1)
        emb = self.elmo_feature_extractor(emb)  # OUT: (B x 32 x L x 1)
        d3_Yhat = self.dssp3_classifier(emb).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 3)
        return d3_Yhat