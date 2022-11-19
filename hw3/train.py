import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.metrics import accuracy_score
from model import *
import matplotlib.pyplot as plt

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

    print(args.model_type)
    print(args.model_type == "attention")
    if args.model_type == "lstm":
        encoder = EncoderLSTM(
            vocab_size=1000,
            pad_idx=maps[0]['<pad>']
        )
        decoder = DecoderLSTM(
            num_actions=len(maps[2]),
            num_targets=len(maps[4]),
            maps=maps
        )
    if args.model_type == "attention":
        encoder = EncoderLSTM(
            vocab_size=1000,
            pad_idx=maps[0]['<pad>']
        )
        decoder = DecoderAttention(
            num_actions=len(maps[2]),
            num_targets=len(maps[4]),
            maps=maps
        )
    if args.model_type == "transformer":
        encoder = EncoderTransformer(
            vocab_size=1000,
            pad_idx=maps[0]['<pad>']
        )
        decoder = DecoderTransformer(
            maps=maps
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
    criterion = torch.nn.CrossEntropyLoss(ignore_index=int(0)) # pad token hard coded
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

    epoch_loss = 0.0
    epoch_acc_num = 0
    epoch_acc_denom = 0

    # iterate over each batch in the dataloader
    for inputs, actions, targets in loader:
        # put model inputs to device
        inputs, actions, targets = inputs.to(device), actions.to(device), targets.to(device) 

        # make predictions
        if args.model_type == "lstm":
            action_dists, target_dists = lstm_inference(
                encoder, decoder, inputs, actions, targets, training, maps, args
            )
        if args.model_type == "attention":
            action_dists, target_dists = attention_inference(
                encoder, decoder, inputs, actions, targets, training, maps, args
            )
        elif args.model_type == "transformer":
            action_dists, target_dists = transformer_inference(
                encoder, decoder, inputs, actions, targets, training, maps, args
            )

        print('show an example prediction')
        print(torch.argmax(action_dists[0], dim=-1))
        print(actions[0])
        print(torch.argmax(target_dists[0], dim=-1))
        print(targets[0])

        # track accuracy
        epoch_acc_denom += torch.sum(actions != maps[2]['<pad>'])
        epoch_acc_denom += torch.sum(targets != maps[4]['<pad>'])
        epoch_acc_num += torch.sum(
            torch.logical_and(
                torch.argmax(action_dists, dim=-1) == actions,
                actions != maps[2]['<pad>']
        ))
        epoch_acc_num += torch.sum(
            torch.logical_and(
                torch.argmax(target_dists, dim=-1) == targets,
                targets != maps[4]['<pad>']
        ))

        # flatten outputs for computing loss
        actions = actions.flatten(0, 1)
        targets = targets.flatten(0, 1)
        action_dists = action_dists.flatten(0, 1)
        target_dists = target_dists.flatten(0, 1)

        # calculate the action and target prediction loss
        action_loss = criterion(action_dists.to(torch.float), actions.to(int))
        target_loss = criterion(target_dists.to(torch.float), targets.to(int))
        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # logging
        print(f"accuracy: {epoch_acc_num / epoch_acc_denom}")
        print(f"loss: {loss.item()}")
        epoch_loss += loss.item()


    epoch_loss /= len(loader)
    epoch_acc = epoch_acc_num / epoch_acc_denom

    return epoch_loss, epoch_acc


def lstm_inference(encoder, decoder, inputs, actions, targets, training, maps, args):
    "Generate a sequence of action-target pairs from an LSTM."
    # encode the instructions
    _, h_final = encoder(inputs)
    h_final = h_final.squeeze()

    # initialize actions and targets with BOS token
    bos_actions = bos_targets = torch.Tensor([2]).expand(h_final.shape[0]).to(int)

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
        if training and args.teacher_forcing:
            action = actions[:,i].to(torch.float)
            target = targets[:,i].to(torch.float)
        else:
            # NOTE: If you get rid of argmax, student forcing probably works better
            action = torch.argmax(action_dist, dim=-1).to(torch.float).squeeze()
            target = torch.argmax(target_dist, dim=-1).to(torch.float).squeeze()

        # get the next prediction
        action_dist, target_dist, h_0, c_0 = decoder(h_0, c_0, action, target)
        action_dists = torch.cat((action_dists, action_dist), dim=1)
        target_dists = torch.cat((target_dists, target_dist), dim=1)
    
    return action_dists, target_dists


def attention_inference(encoder, decoder, inputs, actions, targets, training, maps, args):
    "Generate a sequence of action-target pairs from an LSTM with attention."
    # encode the instructions
    h_seq, h_final = encoder(inputs)
    h_final = h_final.squeeze()

    # initialize actions and targets with BOS token
    bos_actions = bos_targets = torch.Tensor([2]).expand(h_final.shape[0]).to(int)

    # decode the first action-target pair
    action_dist, target_dist, h_0, c_0 = decoder(
        h_0 = h_final.unsqueeze(0),
        c_0 = None,
        action = bos_actions,
        target = bos_targets,
        enc = h_seq,
    )
    action_dists, target_dists = action_dist, target_dist

    # decode the rest autoregressively
    for i in range(actions.shape[1] - 1):
        # teacher forcing
        if training and args.teacher_forcing:
            action = actions[:,i].to(torch.float)
            target = targets[:,i].to(torch.float)
        else:
            # NOTE: If you get rid of one hot, student forcing probably works better
            action = torch.argmax(action_dist, dim=-1).to(torch.float).squeeze()
            target = torch.argmax(target_dist, dim=-1).to(torch.float).squeeze()

        # get the next prediction
        action_dist, target_dist, h_0, c_0 = decoder(
            h_0 = h_0,
            c_0 = c_0,
            action = action,
            target = target,
            enc = h_seq,
        )                
        action_dists = torch.cat((action_dists, action_dist), dim=1)
        target_dists = torch.cat((target_dists, target_dist), dim=1)
    
    return action_dists, target_dists



def transformer_inference(encoder, decoder, inputs, actions, targets, training, maps, args):
    # encode the instructions
    encoded_instructions = encoder(inputs)

    # initialize actions and targets with BOS and CLS token
    bos_tokens = torch.Tensor([maps[2]['<bos>']]).expand(inputs.shape[0]).to(int)
    cls_tokens = torch.Tensor([maps[2]['<cls>']]).expand(inputs.shape[0]).to(int)
    generation = torch.cat((bos_tokens.unsqueeze(1), cls_tokens.unsqueeze(1)), dim=-1)

    # decode the first action-target pair
    action_dist, target_dist = decoder(generation, encoded_instructions)
    action_dists, target_dists = action_dist.unsqueeze(1), target_dist.unsqueeze(1)

    # decode the rest autoregressively
    for i in range(actions.shape[1] - 1):
        # TODO: add len(maps[2]) to each target value
        # teacher forcing
        if training and args.teacher_forcing:
            action = actions[:,i].to(torch.float)
            target = targets[:,i].to(torch.float)
        else:
            # NOTE: If you get rid of one hot, student forcing probably works better
            action = torch.argmax(action_dist, dim=-1).to(torch.float).squeeze()
            target = torch.argmax(target_dist, dim=-1).to(torch.float).squeeze()

        generation = torch.cat((generation, action.unsqueeze(1), target.unsqueeze(1)), dim=-1)

        # get the next prediction
        action_dist, target_dist = decoder(generation, encoded_instructions)   
        action_dist, target_dist = action_dist.unsqueeze(1), target_dist.unsqueeze(1) 

        # append it to the sequence of predictions      
        action_dists = torch.cat((action_dists, action_dist), dim=1)
        target_dists = torch.cat((target_dists, target_dist), dim=1)
    
    return action_dists, target_dists




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

    print()
    print()
    print()
    print()
    print('------ Validation -------')
    print(f"val loss: {val_loss}")
    print(f"val_acc: {val_acc}")
    print()
    print()

    return val_loss, val_acc


def train(args, encoder, decoder, loaders, optimizer, criterion, maps, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    encoder.train()
    decoder.train()

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    train_epoch_nums, val_epoch_nums = [], []

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
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_epoch_nums.append(epoch)

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
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            val_epoch_nums.append(epoch)

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 3 figures for 1) training loss, 2) validation loss, 3) validation accuracy
    # ===================================================== #
    plot_performance(args, train_accs, train_losses,
                     val_accs, val_losses,
                     train_epoch_nums, val_epoch_nums)


def plot_performance(args, train_accs, train_losses,
                     val_accs, val_losses,
                     train_epoch_nums, val_epoch_nums):
    """
    Given the lists of performance tracked in train(),
    shows a matplotlib figure for containing four plots:
        1. Accuracy on Actions 
        2. Accuracy on Targets
        3. Loss on Actions
        4. Loss on Targets
    
    Each plot shows both the train and validation performance
    in order to help assess overfitting on the training data. 
    """

    # Creates four subplots on the same figure, with two lines on each plot
    # The four plots represent accuracy and loss on actions and targets
    # The two lines represent training vs validation measures
    # Credit to: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    fig, axs = plt.subplots(1, 2)

    model_titles = {
        'lstm': "LSTM",
        'attention': "LSTM with Attention",
        'transformer': "Encoder-Decoder Transformer"
    }
    title = model_titles[args.model_type]
    title += " with Teacher Forcing" if args.teacher_forcing else " with Student Forcing"

    axs[0].plot(train_epoch_nums, train_accs, "b--", label="train") 
    axs[0].plot(val_epoch_nums, val_accs, "b-", label="validation")
    axs[0].legend(loc="lower right")
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title(title)

    axs[1].plot(train_epoch_nums, train_losses, "o--", label="train")
    axs[1].plot(val_epoch_nums, val_losses, "o-", label="validation")
    axs[1].legend(loc="lower right")
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].set_title(title)

    plt.show()


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
    parser.add_argument('--teacher_forcing', action='store_true')
    parser.add_argument('--student_forcing', dest='teacher_forcing', action='store_false')

    args = parser.parse_args()

    print(args.teacher_forcing)

    main(args)
