import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.metrics import accuracy_score
from model import DecoderLSTM, EncoderLSTM

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    prefix_match,
    tokenize_words
)

class alfred_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        """
        Initialize the Dataset, which has the same structure as
        the data_dict provided by preprocess_data(), but implements
        the necessary __len__() and __getitem__() methods. 
        """
        super().__init__()
        self.instructions = data_dict['instructions']
        self.actions = data_dict['actions']
        self.targets = data_dict['targets']
    
    def __len__(self):
        """
        Verify that all attributes are of the same length,
        then return that length. 
        """
        assert len(self.instructions) == len(self.actions)
        assert len(self.actions) == len(self.targets)
        return len(self.instructions)

    def __getitem__(self, idx):
        """
        Returns a 3-tuple of the instructions, actions,
        and targets at the idx. 
        """
        return self.instructions[idx], self.actions[idx], self.targets[idx]


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #

    train_dict, val_dict, maps = tokenize_words(args)

    # Store the tokenized data in a custom defined Dataset object
    train_dataset = alfred_dataset(train_dict)
    val_dataset = alfred_dataset(val_dict)

    # Wrap the Dataset objects in iterable DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    return train_loader, val_loader, maps

def setup_model(args, maps, device):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model. Your model should be an
    # an encoder-decoder architecture that encoders the
    # input sentence into a context vector. The decoder should
    # take as input this context vector and autoregressively
    # decode the target sentence. You can define a max length
    # parameter to stop decoding after a certain length.

    # For some additional guidance, you can separate your model
    # into an encoder class and a decoder class.
    # The encoder class forward pass will simply run the input
    # sequence through some recurrent model.
    # The decoder class you will need to implement a teacher
    # forcing mechanism in the forward pass such that instead
    # of feeding the model prediction into the recurrent model,
    # you will give the embedding of the target token.
    # ===================================================== #

    if args.model_type == "lstm":
        encoder = EncoderLSTM(
            vocab_size=1000,
            pad_idx=maps[0]['<pad>']
        )
        decoder = DecoderLSTM(
            num_actions=len(maps[2]),
            num_targets=len(maps[4])
        )

    return encoder, decoder


def setup_optimizer(args, encoder, decoder):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-2)

    return criterion, optimizer


def train_epoch(
    args,
    encoder,
    decoder,
    loader,
    optimizer,
    criterion,
    maps,
    device,
    training=True,
):
    """
    # TODO: implement function for greedy decoding.
    # This function should input the instruction sentence
    # and autoregressively predict the target label by selecting
    # the token with the highest probability at each step.
    # Note this is slightly different from the forward pass of
    # your decoder because you want to pick the token
    # with the highest probability instead of using the
    # teacher-forced token.

    # e.g. Input: "Walk straight, turn left to the counter. Put the knife on the table."
    # Output: [(GoToLocation, diningtable), (PutObject, diningtable)]
    # Also write some code to compute the accuracy of your
    # predictions against the ground truth.
    """

    epoch_loss = 0.0
    epoch_acc_numerator = 0
    epoch_acc_denominator = 0

    # iterate over each batch in the dataloader
    for inputs, actions, targets in loader:
        # put model inputs to device
        inputs, actions, targets = inputs.to(device), actions.to(device), targets.to(device) 

        # encode the instructions
        h_seq, h_final = encoder(inputs)
        h_final = h_final.squeeze()

        # initialize actions and targets with BOS token
        bos_tokens = torch.Tensor([2]).expand(h_final.shape[0]).to(int)
        bos_actions = F.one_hot(bos_tokens, len(maps[2]))
        bos_targets = F.one_hot(bos_tokens, len(maps[4]))

        # decode the first action-target pair
        action_dist, target_dist, h_0, c_0 = decoder(
            h_0 = h_final.unsqueeze(0),
            c_0 = None,
            action = bos_actions,
            target = bos_targets
        )
        action_dists, target_dists = action_dist, target_dist

        # decode the rest autoregressively
        for i in range(actions.shape[1] - 1):
            # teacher forcing
            if training == True:
                action = actions[:,i,:].to(torch.float)
                target = targets[:,i,:].to(torch.float)
            else:
                # NOTE: If you get rid of one hot, student forcing probably works better
                action = F.one_hot(torch.argmax(action_dist, dim=-1), len(maps[2])).to(torch.float).squeeze()
                target = F.one_hot(torch.argmax(target_dist, dim=-1), len(maps[4])).to(torch.float).squeeze()

            # get the next prediction
            action_dist, target_dist, h_0, c_0 = decoder(h_0, c_0, action, target)
            action_dists = torch.cat((action_dists, action_dist), dim=1)
            target_dists = torch.cat((target_dists, target_dist), dim=1)

        # calculate the action and target prediction loss
        action_loss = criterion(action_dists.to(torch.float), actions.to(torch.float))
        target_loss = criterion(target_dists.to(torch.float), targets.to(torch.float))
        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # logging
        
        epoch_loss += loss.item()

    epoch_loss /= len(loader)

    return epoch_loss, epoch_acc


def validate(args, encoder, decoder, loader, optimizer, criterion, maps, device):
    # set model to eval mode
    encoder.eval()
    decoder.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            encoder,
            decoder,
            loader,
            optimizer,
            criterion,
            maps,
            device,
            training=False,
        )

    return val_loss, val_acc


def train(args, encoder, decoder, loaders, optimizer, criterion, maps, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    encoder.train()
    decoder.train()

    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        train_loss, train_acc = train_epoch(
            args,
            encoder,
            decoder,
            loaders["train"],
            optimizer,
            criterion,
            maps,
            device,
        )

        # some logging
        print(f"train loss : {train_loss}")

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                args,
                encoder,
                decoder,
                loaders["val"],
                optimizer,
                criterion,
                maps,
                device,
            )

            print(f"val loss : {val_loss} | val acc: {val_acc}")

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 3 figures for 1) training loss, 2) validation loss, 3) validation accuracy
    # ===================================================== #




def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, maps = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    encoder, decoder = setup_model(args, maps, device)
    print(encoder)
    print(decoder)

    # get optimizer and loss functions
    criterion, optimizer = setup_optimizer(args, encoder, decoder)

    if args.eval:
        val_loss, val_acc = validate(
            args,
            encoder,
            decoder,
            loaders["val"],
            optimizer,
            criterion,
            maps,
            device,
        )
    else:
        train(args, encoder, decoder, loaders, optimizer, criterion, maps, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, type=int, help="number of epochs between every eval loop"
    )
    parser.add_argument("--model_type", type=str, help="encoder decoder model to run")


    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()

    main(args)
