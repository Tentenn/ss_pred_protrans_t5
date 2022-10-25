"""
Adapted from Michael Heinzinger Notebook
https://colab.research.google.com/drive/1TUj-ayG3WO52n5N50S7KH9vtt6zRkdmj?usp=sharing
"""
from transformers import BertModel
import torch

# Convolutional neural network (two convolutional layers) to predict secondary structure
class BertCNN(torch.nn.Module):
    def __init__(self):
        super(BertCNN, self).__init__()
        
        self.bert = BertModel.from_pretrained("Rostlab/prot_bert")
        
        # This is only called "elmo_feature_extractor" for historic reason
        # CNN weights are trained on ProtT5 embeddings
        self.elmo_feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
        )
        n_final_in = 32
        self.dssp3_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 3, kernel_size=(7, 1), padding=(3, 0))  # 7
        )

    def forward(self, input_ids):
        # create embeddings
        emb = self.bert(input_ids).last_hidden_state
        # trim last
        emb = emb[:, 1:-1, :]
        
        x = emb
        
        # print("emb shape", emb.shape)
        
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        x = self.elmo_feature_extractor(x)  # OUT: (B x 32 x L x 1)
        d3_Yhat = self.dssp3_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 3)
        return d3_Yhat