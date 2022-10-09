from tqdm import tqdm
import torch
import torch.nn.functional as F
import argparse
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import re

from model import LSTM
from utils import get_device, tokenize_words
from bpe import tokenize_bpe


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
    # Process the text into tokens stored in dictionaries
    if args.bpe:
        train_dict, val_dict, maps = tokenize_bpe(args)
    else:
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

    model = LSTM(args, maps)
    return model


def setup_optimizer(args, model):
    """
    return:
        - action_criterion: loss_fn
        - target_criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    action_criterion = torch.nn.CrossEntropyLoss()
    target_criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    return action_criterion, target_criterion, optimizer

def train_epoch(
    args,
    model,
    loader,
    optimizer,
    action_criterion,
    target_criterion,
    device,
    training=True,
):
    # initialize epoch loss
    epoch_action_loss = 0.0
    epoch_target_loss = 0.0

    # keep track of the model predictions for computing accuracy
    action_preds = []
    target_preds = []
    action_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    for inputs, actions, targets in tqdm(loader):
        # put model inputs to device
        inputs, actions, targets = inputs.to(device), actions.to(device), targets.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        actions_out, targets_out = model(inputs)

        # calculate the action and target prediction loss
        action_loss = action_criterion(actions_out.squeeze(), actions.float())
        target_loss = target_criterion(targets_out.squeeze(), targets.float())

        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_action_loss += action_loss.item()
        epoch_target_loss += target_loss.item()

        # take the prediction with the highest probability
        # NOTE: this could change depending on if you apply Sigmoid in your forward pass
        action_preds_ = actions_out.argmax(-1)
        target_preds_ = targets_out.argmax(-1)
        actions_ = actions.argmax(-1)
        targets_ = targets.argmax(-1)

        # aggregate the batch predictions + labels
        action_preds.extend(action_preds_.cpu().numpy())
        target_preds.extend(target_preds_.cpu().numpy())
        action_labels.extend(actions_.cpu().numpy())
        target_labels.extend(targets_.cpu().numpy())

    action_acc = accuracy_score(action_preds, action_labels)
    target_acc = accuracy_score(target_preds, target_labels)

    return epoch_action_loss, epoch_target_loss, action_acc, target_acc


def validate(
    args, model, loader, optimizer, action_criterion, target_criterion, device
):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():

        val_action_loss, val_target_loss, action_acc, target_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )

    return val_action_loss, val_target_loss, action_acc, target_acc


def train(args, model, loaders, optimizer, action_criterion, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()
    model.to(device)

    # Setup lists to track loss and accuracy over epochs
    if args.show_plot==True:
        train_action_accs, train_action_losses = [],[]
        train_target_accs, train_target_losses = [],[]
        val_action_accs, val_action_losses = [],[]
        val_target_accs, val_target_losses = [],[]
        train_epoch_nums, val_epoch_nums = [],[]

    for epoch in tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        (
            train_action_loss,
            train_target_loss,
            train_action_acc,
            train_target_acc,
        ) = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )

        # some logging
        print(f"train action loss : {train_action_loss}")
        print(f"train target loss: {train_target_loss}")
        print(f"train action acc : {train_action_acc}")
        print(f"train target acc: {train_target_acc}")
        if args.show_plot==True:
            train_action_accs.append(train_action_acc)
            train_action_losses.append(train_action_loss)
            train_target_accs.append(train_target_acc)
            train_target_losses.append(train_target_loss)
            train_epoch_nums.append(epoch)


        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
            )

            print(
                f"val action loss : {val_action_loss} | val target loss: {val_target_loss}"
            )
            print(
                f"val action acc : {val_action_acc} | val target losaccs: {val_target_acc}"
            )
            if args.show_plot==True:
                val_action_accs.append(val_action_acc)
                val_action_losses.append(val_action_loss)
                val_target_accs.append(val_target_acc)
                val_target_losses.append(val_target_loss)
                val_epoch_nums.append(epoch)

    if args.show_plot==True:
        # Divides Loss by len(loss) to find Average Loss Per Example
        train_action_losses = [i/len(loaders['train']) for i in train_action_losses]
        train_target_losses = [i/len(loaders['train']) for i in train_target_losses]
        val_action_losses = [i/len(loaders['val']) for i in val_action_losses]
        val_target_losses = [i/len(loaders['val']) for i in val_target_losses]

        # Generates the plot
        plot_performance(train_action_accs, train_action_losses,
                         train_target_accs, train_target_losses,
                         val_action_accs, val_action_losses,
                         val_target_accs, val_target_losses,
                         train_epoch_nums, val_epoch_nums)


def plot_performance(train_action_accs, train_action_losses,
                     train_target_accs, train_target_losses,
                     val_action_accs, val_action_losses,
                     val_target_accs, val_target_losses,
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
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(train_epoch_nums, train_action_accs, "b--", label="train") 
    axs[0, 0].plot(val_epoch_nums, val_action_accs, "b-", label="validation")
    axs[0, 0].legend(loc="lower right")
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_ylim([0, 1])
    axs[0, 0].set_title('Accuracy on Actions')

    axs[0, 1].plot(train_epoch_nums, train_target_accs, "o--", label="train")
    axs[0, 1].plot(val_epoch_nums, val_target_accs, "o-", label="validation")
    axs[0, 1].legend(loc="lower right")
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].set_ylim([0, 1])
    axs[0, 1].set_title('Accuracy on Targets')

    axs[1, 0].plot(train_epoch_nums, train_action_losses, "m--", label="train")
    axs[1, 0].plot(val_epoch_nums, val_action_losses, "m-", label="validation")
    axs[1, 0].legend(loc="upper right")
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].set_title('Loss on Actions')

    axs[1, 1].plot(train_epoch_nums, train_target_losses, "g--", label="train")
    axs[1, 1].plot(val_epoch_nums, val_target_losses, "g-", label="validation")
    axs[1, 1].legend(loc="upper right")
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].set_title('Loss on Targets')
    
    plt.show()


def main(args):
    device = get_device(args.force_cpu)
    torch.manual_seed(42)

    # get dataloaders
    train_loader, val_loader, maps = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, maps, device)
    print(model)

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(
            args, model, loaders, optimizer, action_criterion, target_criterion, device
        )

def alfred_args():
    """
    Parses command-line arguments for the model. 
    Returns args of type ArgumentParser
    """
    # Initialize
    parser = argparse.ArgumentParser()
    
    # Add arguments
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument("--model_output_dir", type=str, 
                        help="where to save model outputs")
    parser.add_argument("--batch_size", type=int, 
                        default=32, help="size of each batch in loader")
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", default=1000, 
                        help="number of training epochs", type=int)
    parser.add_argument("--val_every", default=5, 
                        help="number of epochs between every eval loop", type=int)
    parser.add_argument("--vocab_size", default=1000, 
                        help="number of tokens in vocab", type=int)
    parser.add_argument("--maxpool", action="store_true", 
                        help="model uses lstm hidden state and token outputs")
    parser.add_argument('--show_plot', action='store_true', 
                        help='displays plot of performance over epochs')
    parser.add_argument('--bpe', action='store_true', 
                    help='uses BPE instead of word-level tokenization')

    # Parse
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = alfred_args()
    main(args)