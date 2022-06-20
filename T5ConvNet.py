"""
Adapted and modified from Michael Heinzinger Notebook
https://colab.research.google.com/drive/1TUj-ayG3WO52n5N50S7KH9vtt6zRkdmj?usp=sharing
"""

from transformers import T5EncoderModel, T5Tokenizer
import torch

class T5CNN(torch.nn.Module):
    def __init__(self):
        super(T5CNN, self).__init__()

        self.t5 = T5EncoderModel.from_Pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
        self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")

        self.elmo_feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
        )
        n_final_in = 32
        self.dssp3_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 3, kernel_size=(7, 1), padding=(3, 0))  # 7
        )


    def encode(self):
        ## Encode sequence like in signalp6 or in code
        return ids,

    def forward(self, x):
        # IN: X = (B x L x F);
        # OUT: (B x F x L, 1)
        x = x.permute(0, 2, 1).unsqueeze(dim=-1) ## ???

        """
        Create embeddings 
        """

        """
        backward function
        """
        d3_Yhat = self.dssp3_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 3)