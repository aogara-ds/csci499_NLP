import torch
import torch.nn.functional as F


class SkipGram(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        # Set the hyperparameters
        self.embedding_dim = 128
        self.vocab_size = args.vocab_size

        # Initialize the layers
        self.embed = torch.nn.Embedding(
            num_embeddings = self.vocab_size,
            embedding_dim = self.embedding_dim,
            padding_idx = 0
        )
        self.projection = torch.nn.Linear(
            in_features = self.embedding_dim,
            out_features = self.vocab_size
        )
    
    def forward(self, inputs):
        # Pass the inputs through the layers
        embeds = self.embed(inputs)
        outputs = self.projection(embeds)
        
        return outputs
    
    def train(self, mode: bool = True):
        self.training = mode
    
    def eval(self, mode: bool = False):
        self.training = mode