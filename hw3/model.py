import torch
import torch.nn as nn


class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, pad_idx):
        super().__init__()

        # Initialize some hyperparameters
        self.embedding_dim = 7
        self.lstm_dim = 11
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
    def __init__(self, num_actions, num_targets):
        super().__init__()

        # Initialize settings
        self.input_dim = 16
        self.num_actions = num_actions
        self.num_targets = num_targets
        
        # Initialize embedding layers
        self.embed_bos = nn.Embedding(
            num_embeddings = 1,
            embedding_dim = self.input_dim
        )
        self.embed_action = nn.Embedding(
            num_embeddings = self.num_actions,
            embedding_dim = int(self.input_dim / 2)
        )
        self.embed_target = nn.Embedding(
            num_embeddings = self.num_targets,
            embedding_dim = int(self.input_dim / 2)
        )

        # Initialize LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_dim, 
            hidden_size=self.input_dim,
            num_layers=1,
            batch_first=True
        )

        # Initialize fully-connected linear layers
        self.fc_action = torch.nn.Linear(in_features=self.input_dim,
                                         out_features=self.num_actions)
        self.fc_target = torch.nn.Linear(in_features=self.input_dim,
                                         out_features=self.num_targets)


    def forward(self, h_0, c_0, action, target, bos):
        if bos == True:
            # Begin decoding with a learnable BOS token
            x_0 = self.embed_bos(0)
        else: 
            # Embed the action and target inputs
            a_0 = self.embed_action(action)
            o_0 = self.embed_target(target)
            # TODO: Check that this concat is correct
            x_0 = torch.cat(a_0, o_0)
        
        # Run the LSTM to generate a single embedding
        x_1, (h_1, c_1) = self.lstm(x_0, h_0, c_0)

        # Predict action and target
        action_dist = self.fc_action(x_1)
        target_dist = self.fc_target(x_1)

        return action_dist, target_dist, h_1, c_1


# class EncoderDecoder(nn.Module):
#     """
#     Wrapper class over the Encoder and Decoder.
#     TODO: edit the forward pass arguments to suit your needs
#     """

#     def __init__(self, model_type, maps):
#         if model_type == "lstm":
#             self.encoder = EncoderLSTM(
#                 vocab_size=1000,
#                 pad_idx=maps[0]['<pad>']
#             )
#             self.decoder = DecoderLSTM(
#                 num_actions=len(maps[2]),
#                 num_targets=len(maps[4])
#             )
        
#     def forward(self, x):

#         pass
