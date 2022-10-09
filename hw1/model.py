import torch
import torch.nn.functional as F

class LSTM(torch.nn.Module):
    def __init__(self, args, maps):
        """
        Defines our custom LSTM class. 
        """
        # The internet says I should initialize my parent class?
        # This seems silly, shouldn't Python take care of that?
        # Not sure, including it anyways. See here: 
        # https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods
        super().__init__()

        # Initialize some hyperparameters
        self.embedding_dim = 128
        self.lstm_dim = 128
        self.maxpool = args.maxpool
        if args.maxpool:
            self.lstm_dim = int(self.lstm_dim / 2)
            self.final_hidden_dim = self.lstm_dim * 2
        else:
            self.final_hidden_dim = self.lstm_dim
        self.lstm_layers = 1
        self.dropout = 0
        self.bidirectional = False
        self.action_classes = len(maps[2])
        self.target_classes = len(maps[4])

        # Initialize an embedding layer the size of our vocabulary
        self.embed = torch.nn.Embedding(num_embeddings=args.vocab_size, 
                                        embedding_dim=self.embedding_dim,
                                        padding_idx=maps[0]['<pad>'])

        # Initialize an LSTM block using our hyperparameters
        self.lstm = torch.nn.LSTM(input_size=self.embedding_dim, 
                                  hidden_size=self.lstm_dim,
                                  num_layers=self.lstm_layers,
                                  dropout=self.dropout,
                                  bidirectional=self.bidirectional,
                                  batch_first=True)

        # Initialize a fully-connected linear layer
        self.fc_action = torch.nn.Linear(in_features=self.final_hidden_dim,
                                         out_features=self.action_classes)
        self.fc_target = torch.nn.Linear(in_features=self.final_hidden_dim,
                                         out_features=self.target_classes)

    def forward(self, inputs):
        """
        Performs a forward pass through the LSTM. 
        """
        # Embed the inputs
        embeds = self.embed(inputs)

        # Run the embeddings through the LSTM
        lstm_output, (h_n, c_n)  = self.lstm(embeds)

        # Maxpool the LSTM word embeddings and concatenate with the hidden state
        if self.maxpool:
            max_lstm_output = F.max_pool1d(lstm_output.transpose(-1, -2), 
                                        kernel_size=lstm_output.size()[1])
            h_n = torch.cat((max_lstm_output.squeeze(), h_n.squeeze()), dim=-1)

        # Use two separate fully connected layers to predict actions and targets
        action_mass = self.fc_action(h_n).squeeze()
        target_mass = self.fc_target(h_n).squeeze()

        return action_mass, target_mass