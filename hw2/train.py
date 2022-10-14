import argparse
import os
import tqdm
import torch
from sklearn.model_selection import train_test_split
from torchmetrics import JaccardIndex

from eval_utils import downstream_validation
import utils
import data_utils
from model import SkipGram
import numpy as np

import matplotlib.pyplot as plt

class books_dataset(torch.utils.data.Dataset):
    def __init__(self, tokens, labels):
        """
        Initialize the Dataset which implements
        the __len__() and __getitem__() methods. 
        """
        super().__init__()
        assert len(tokens) == len(labels)
        self.tokens = tokens
        self.labels = labels
    
    def __len__(self):
        """
        Verify that all attributes are of the same length,
        then return that length. 
        """
        assert len(self.tokens) == len(self.labels)
        return len(self.tokens)

    def __getitem__(self, idx):
        """
        Returns a 3-tuple of the instructions, actions,
        and targets at the idx. 
        """
        return self.tokens[idx], self.labels[idx]

def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """

    # read in training data from books dataset
    sentences = data_utils.process_book_dir(args.books_dir)

    # build one hot maps for input and output
    (
        vocab_to_index,
        index_to_vocab,
        suggested_padding_len,
    ) = data_utils.build_tokenizer_table(sentences, vocab_size=args.vocab_size)

    # create encoded input and output numpy matrices for the entire dataset and then put them into tensors
    encoded_sentences, lens = data_utils.encode_data(
        sentences,
        vocab_to_index,
        suggested_padding_len,
    )

    tokens, labels = data_utils.get_tokens_and_labels(encoded_sentences, lens, args)

    (
        train_tokens, 
        val_tokens, 
        train_labels, 
        val_labels
    ) = train_test_split(tokens, labels, test_size = 0.2)

    train_data = books_dataset(train_tokens, train_labels)
    val_data = books_dataset(val_tokens, val_labels)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size)

    return train_loader, val_loader, index_to_vocab


def setup_model(args):
    model = SkipGram(args)
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
    return criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    iou_fn,
    training=True
):
    model.train()
    epoch_loss = 0.0
    batch_iou_list = []

    # iterate over each batch in the dataloader
    for (inputs, labels) in tqdm.tqdm(loader):
        # put model inputs to device
        inputs, labels = inputs.to(device).long(), labels.to(device).long()

        # calculate the loss and train IOU and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        pred_logits = model(inputs)

        # calculate prediction loss
        loss = criterion(pred_logits.squeeze().float(), labels.float())

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_loss += loss.item()

        # compute metrics
        preds = torch.topk(pred_logits, k=4, dim=-1)[-1]      
        label_idxs = torch.topk(labels, k=4, dim=-1)[-1]
        batch_iou = iou_fn(preds, label_idxs)
        batch_iou_list.append(batch_iou)

    # track epoch metrics
    iou = sum(batch_iou_list) / len(batch_iou_list)
    epoch_loss /= len(loader)

    return epoch_loss, iou


def validate(args, model, loader, optimizer, criterion, device, iou_fn):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_iou = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            iou_fn,
            training=False,
        )

    return val_loss, val_iou

def plot_performance(train_ious, train_losses,
                     val_ious, val_losses,
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
    axs[0, 0].plot(train_epoch_nums, train_losses) 
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('BCE Loss')
    axs[0, 0].set_title('Training Loss')

    axs[0, 1].plot(train_epoch_nums, train_ious)
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Intersection over Union (IOU)')
    axs[0, 1].set_title('Training Intersection over Union (IOU)')

    axs[1, 0].plot(val_epoch_nums, val_losses)
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('BCE Loss')
    axs[1, 0].set_title('Validation Loss')

    axs[1, 1].plot(val_epoch_nums, val_ious)
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Intersection over Union (IOU)')
    axs[1, 1].set_title('Validation Intersection over Union (IOU)')
    
    plt.show()


def main(args):
    device = utils.get_device(args.force_cpu)

    # load analogies for downstream eval
    external_val_analogies = utils.read_analogies(args.analogies_fn)

    if args.downstream_eval:
        word_vec_file = os.path.join(args.output_dir, args.word_vector_fn)
        assert os.path.exists(word_vec_file), "need to train the word vecs first!"
        downstream_validation(word_vec_file, external_val_analogies)
        return

    # get dataloaders
    train_loader, val_loader, i2v = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args)
    print(model)

    # get optimizer
    criterion, optimizer = setup_optimizer(args, model)

    # initialize iou function
    iou_fn = JaccardIndex(args.vocab_size)

    # initialize performance tracking
    train_ious = []
    train_losses = [] 
    val_ious = [] 
    val_losses = [] 
    train_epoch_nums = [] 
    val_epoch_nums = []

    for epoch in range(args.num_epochs):
        # train model for a single epoch
        print(f"Epoch {epoch}")
        train_loss, train_iou = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
            iou_fn
        )

        print(f"train loss : {train_loss} | train iou: {train_iou}")
        train_ious.append(train_iou)
        train_losses.append(train_loss)
        train_epoch_nums.append(epoch)

        if epoch % args.val_every == 0:
            val_loss, val_iou = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
                iou_fn
            )
            print(f"val loss : {val_loss} | val iou: {val_iou}")
            val_ious.append(val_iou)
            val_losses.append(val_loss)
            val_epoch_nums.append(epoch)

            # ======================= NOTE ======================== #
            # Saving the word vectors to disk and running the eval
            # can be costly when you do it multiple times. You could
            # change this to run only when your training has concluded.
            # However, incremental saving means if something crashes
            # later or you get bored and kill the process you'll still
            # have a word vector file and some results.
            # ===================================================== #

            # save word vectors
            word_vec_file = os.path.join(args.output_dir, args.word_vector_fn)
            print("saving word vec to ", word_vec_file)
            utils.save_word2vec_format(word_vec_file, model, i2v)

            # evaluate learned embeddings on a downstream task
            downstream_validation(word_vec_file, external_val_analogies)


        if epoch % args.save_every == 0:
            ckpt_file = os.path.join(args.output_dir, "model.ckpt")
            print("saving model to ", ckpt_file)
            torch.save(model, ckpt_file)
    
    plot_performance(train_ious, train_losses,
                     train_ious, val_losses,
                     train_epoch_nums, val_epoch_nums)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default='wordvecs/', type=str, help="where to save training outputs")
    parser.add_argument("--books_dir", type=str, help="where the book dataset is stored")
    parser.add_argument(
        "--downstream_eval",
        action="store_true",
        help="run downstream eval on trained word vecs",
    )
    # ======================= NOTE ======================== #
    # If you adjust the vocab_size down below 3000, there 
    # may be analogies in the downstream evaluation that have
    # words that are not in your vocabulary, resulting in
    # automatic (currently) zero score for an ABCD where one
    # of A, B, C, or D is not in the vocab. A visible warning
    # will be generated by the evaluation loop for these examples.
    # ===================================================== #
    parser.add_argument(
        "--vocab_size", type=int, default=3000, help="size of vocabulary"
    )
    parser.add_argument(
        "--context_size", type=int, default=2, help="number of tokens to look forwards and back"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument(
        "--analogies_fn", type=str, help="filepath to the analogies json file"
    )
    parser.add_argument(
        "--word_vector_fn", type=str, help="filepath to store the learned word vectors",
        default='learned_word_vectors.txt'
    )
    parser.add_argument(
        "--num_epochs", default=5, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--val_every",
        default=1,
        type=int,
        help="number of epochs between every eval loop",
    )
    parser.add_argument(
        "--save_every",
        default=5,
        type=int,
        help="number of epochs between saving model checkpoint",
    )

    args = parser.parse_args()
    main(args)
