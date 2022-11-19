import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

# ========================================== #
#               Vanilla LSTM                 #
# ========================================== #

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, pad_idx):
        super().__init__()

        # Initialize some hyperparameters
        self.embedding_dim = 16
        self.lstm_dim = 16
        self.lstm_layers = 1
        self.dropout = 0
        self.bidirectional = False

        # Initialize an embedding layer the size of our vocabulary
        self.embed = nn.Embedding(num_embeddings=vocab_size, 
                                        embedding_dim=self.embedding_dim,
                                        padding_idx=pad_idx)

        # Initialize an LSTM block using our hyperparameters
        self.lstm = nn.LSTM(input_size=self.embedding_dim, 
                                  hidden_size=self.lstm_dim,
                                  num_layers=self.lstm_layers,
                                  dropout=self.dropout,
                                  bidirectional=self.bidirectional,
                                  batch_first=True)

    def forward(self, inputs):
        # Embed the inputs
        embeds = self.embed(inputs)
        # Run the embeddings through the LSTM
        h_seq, (h_final, _)  = self.lstm(embeds)
        # h_seq has a dimension of sequence length, h_final does not
        return h_seq, h_final


class DecoderLSTM(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.

    Given an encoder hidden state and two input tokens, emit two output tokens
    """
    def __init__(self, num_actions, num_targets, maps):
        super().__init__()

        # Initialize settings
        self.embedding_dim = 16
        self.num_actions = num_actions
        self.num_targets = num_targets
        
        # Initialize embedding layers
        self.embed_action = nn.Embedding(
            num_embeddings=self.num_actions, 
            embedding_dim=int(self.embedding_dim / 2),
            padding_idx=maps[2]['<pad>']
        )
        self.embed_target = nn.Embedding(
            num_embeddings=self.num_targets, 
            embedding_dim=int(self.embedding_dim / 2),
            padding_idx=maps[4]['<pad>']
        )

        # Initialize LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim, 
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True
        )

        # Initialize fully-connected linear layers
        self.fc_action = torch.nn.Linear(in_features=self.embedding_dim,
                                         out_features=self.num_actions)
        self.fc_target = torch.nn.Linear(in_features=self.embedding_dim,
                                         out_features=self.num_targets)


    def forward(self, h_0, c_0, action, target):
        # Embed the action and target inputs
        a_0 = self.embed_action(action.to(torch.int))
        o_0 = self.embed_target(target.to(torch.int))
        x_0 = torch.cat((a_0, o_0), dim=1)

        # Initialize hidden state if necessary
        if c_0 == None:
            c_0 = torch.zeros((h_0.shape[1], self.embedding_dim)).unsqueeze(0)

        # unsqueeze the input tokens
        x_0 = x_0.unsqueeze(1)

        # Run the LSTM to generate a single embedding
        x_1, (h_1, c_1) = self.lstm(x_0, (h_0, c_0))
        
        # Predict action and target
        action_dist = self.fc_action(x_1)
        target_dist = self.fc_target(x_1)

        return action_dist, target_dist, h_1, c_1


# ========================================== #
#         LSTM + Attention Decoder           #
# ========================================== #


class DecoderAttention(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.

    Given an encoder hidden state and two input tokens, emit two output tokens
    """
    def __init__(self, num_actions, num_targets, maps):
        super().__init__()

        # Initialize settings
        self.embedding_dim = 16
        self.num_actions = num_actions
        self.num_targets = num_targets
        
        # Initialize embedding layers
        self.embed_action = nn.Embedding(
            num_embeddings=self.num_actions, 
            embedding_dim=int(self.embedding_dim / 2),
            padding_idx=maps[2]['<pad>']
        )
        self.embed_target = nn.Embedding(
            num_embeddings=self.num_targets, 
            embedding_dim=int(self.embedding_dim / 2),
            padding_idx=maps[4]['<pad>']
        )

        # Initialize LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim, 
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=False
        )

        # Initialize fully-connected linear layers
        self.fc_action = torch.nn.Linear(in_features=self.embedding_dim,
                                         out_features=self.num_actions)
        self.fc_target = torch.nn.Linear(in_features=self.embedding_dim,
                                         out_features=self.num_targets)
        
        # Initialize Multihead Attention
        self.attention = torch.nn.MultiheadAttention(self.embedding_dim, 1)
        self.project_query = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.project_key = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.project_value = torch.nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, h_0, c_0, action, target, enc):
        # Embed the action and target inputs
        a_0 = self.embed_action(action.to(torch.int))
        o_0 = self.embed_target(target.to(torch.int))
        x_0 = torch.cat((a_0, o_0), dim=-1)

        # Initialize hidden state if necessary
        if c_0 == None:
            c_0 = torch.zeros((h_0.shape[1], self.embedding_dim)).unsqueeze(0)

        # unsqueeze the input tokens
        x_0 = x_0.unsqueeze(1)

        # hidden state attends to encoder sequence
        enc = enc.transpose(0, 1)
        query = self.project_query(h_0)
        key = self.project_key(enc)
        value = self.project_value(enc)
        attention_output, _ = self.attention(query, key, value)

        # Run the LSTM to generate a single embedding
        x_1, (h_1, c_1) = self.lstm(attention_output, (h_0, c_0))
        
        # Predict action and target
        action_dist = self.fc_action(x_1)
        target_dist = self.fc_target(x_1)

        # transpose to put batch first
        action_dist = action_dist.transpose(0,1)
        target_dist = target_dist.transpose(0,1)

        return action_dist, target_dist, h_1, c_1



# ========================================== #
#                Transformer                 #
# ========================================== #


class EncoderTransformer(nn.Module):
    def __init__(self, vocab_size=1000, pad_idx=2):
        # Initialize nn.Module
        super().__init__()

        # Set hyperparameters
        self.embedding_dim = 16
        self.attention_blocks = 2

        # Embedding Layer
        self.embed = nn.Embedding(
            num_embeddings = vocab_size, 
            embedding_dim = self.embedding_dim,
            padding_idx = pad_idx, 
        )

        # Positional Encoding
        self.pe = PositionalEncoding(d_model=self.embedding_dim)

        # Transformer Blocks
        self.blocks = [EncoderBlock() for _ in range(self.attention_blocks)]


    def forward(self, x):
        # Embed
        x = self.embed(x)

        # Positional Encoding
        x = self.pe(x)

        # Attention Blocks
        for i in range(self.attention_blocks):
            x = self.blocks[i](x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()

        # Set hyperparameters
        self.hidden_dim = 16
        self.qkv_dim = 16
        self.ffnn_dim = 16

        # Projection Layer
        self.project_query = nn.Linear(self.hidden_dim, self.qkv_dim)
        self.project_key = nn.Linear(self.hidden_dim, self.qkv_dim)
        self.project_value = nn.Linear(self.hidden_dim, self.qkv_dim)

        # FFNN
        self.ff1 = nn.Linear(self.qkv_dim, self.ffnn_dim)
        self.ff2 = nn.Linear(self.ffnn_dim, self.hidden_dim)

    def forward(self, x):
        # Projection
        q = self.project_query(x)
        k = self.project_key(x)
        v = self.project_value(x)

        # Scaled Dot Product Attention
        scores = F.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.qkv_dim**(1/2)), dim=-1)
        attention_output = torch.matmul(scores, v)
        
        # Residual and Layer Norm
        x = F.layer_norm(x + attention_output, x.shape)

        # FFNN
        ff_output = self.ff1(x)
        ff_output = F.relu(ff_output)
        ff_output = self.ff2(ff_output)

        # Residual and Layer Norm
        x = F.layer_norm(x + ff_output, x.shape)

        return x


class DecoderTransformer(nn.Module):
    def __init__(self, maps):
        super().__init__()

        # Set hyperparameters
        self.embedding_dim = 16
        self.attention_blocks = 2

        # Embedding Layer
        self.embed = nn.Embedding(
            num_embeddings = len(maps[2]) + len(maps[4]), 
            embedding_dim = self.embedding_dim,
            padding_idx = maps[2]['<pad>'], 
        )

        # Positional Encoding
        self.pe = PositionalEncoding(d_model=self.embedding_dim)

        # Transformer Blocks
        self.blocks = [DecoderBlock() for _ in range(self.attention_blocks)]

        # Classification Heads
        self.fc_action = torch.nn.Linear(in_features=self.embedding_dim,
                                         out_features=len(maps[2]))
        self.fc_target = torch.nn.Linear(in_features=self.embedding_dim,
                                         out_features=len(maps[4]))

    def forward(self, x, enc):
        # Embed
        x = self.embed(x.to(int))

        # Positional Encoding
        x = self.pe(x)

        # Attention Blocks
        for i in range(self.attention_blocks):
            x = self.blocks[i](x, enc)
        
        # Extract the final two tokens in the sequence
        x_action = x[:, -2]
        x_target = x[:, -1]

        # TODO: Extract the hidden state of the final CLS token
        action_dist = self.fc_action(x_action)
        target_dist = self.fc_target(x_target)

        return action_dist, target_dist


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()

        # Set hyperparameters
        self.hidden_dim = 16
        self.qkv_dim = 16
        self.ffnn_dim = 16

        # Projection Layers
        self.project_query = [nn.Linear(self.hidden_dim, self.qkv_dim) for _ in range(2)]
        self.project_key = [nn.Linear(self.hidden_dim, self.qkv_dim) for _ in range(2)]
        self.project_value = [nn.Linear(self.hidden_dim, self.qkv_dim) for _ in range(2)]

        # FFNN
        self.ff1 = nn.Linear(self.qkv_dim, self.ffnn_dim)
        self.ff2 = nn.Linear(self.ffnn_dim, self.hidden_dim)

    def forward(self, x, enc):
        # First Projection
        q = self.project_query[0](x)
        k = self.project_key[0](x)
        v = self.project_value[0](x)

        # First Scaled Dot Product Attention
        scores = F.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.qkv_dim**(1/2)), dim=-1)
        attention_output = torch.matmul(scores, v)
        
        # First Residual and Layer Norm
        x = F.layer_norm(x + attention_output, x.shape)

        # Second Projection -- Cross Attention
        q = self.project_query[1](x)
        k = self.project_key[1](enc)
        v = self.project_value[1](enc)

        # First Scaled Dot Product Attention
        scores = F.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.qkv_dim**(1/2)), dim=-1)
        attention_output = torch.matmul(scores, v)

        # Second Residual and Layer Norm
        x = F.layer_norm(x + attention_output, x.shape)

        # FFNN
        ff_output = self.ff1(x)
        ff_output = F.relu(ff_output)
        ff_output = self.ff2(ff_output)

        # Residual and Layer Norm
        x = F.layer_norm(x + ff_output, x.shape)

        return x


class PositionalEncoding(nn.Module):
    # This class is copied from The Annotated Transformer
    # https://nlp.seas.harvard.edu/2018/04/03/attention.html
    # Everything else I wrote myself using only Vaswani et al., 2017
    def __init__(self, d_model, max_len=300):
        super(PositionalEncoding, self).__init__()
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return x