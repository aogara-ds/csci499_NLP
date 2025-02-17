import re
import torch
import numpy as np
from collections import Counter
import time
import json
import torch.nn.functional as F

def tokenize_words(args):
    """
    Word-level tokenizer. Takes args as input, 
    returns encoded data and encoder maps. 
    """
    # Code for Tokenizer Analysis
    unk_count, word_count = 0, 0
    begin_tokenizer = time.time()

    # Load the data from the json
    data = json.load(open(args.in_data_fn))

    # Use the utils to construct the necessary maps
    vocab_to_index, index_to_vocab, max_seq_len, max_output_len = (
        build_tokenizer_table(data['train'])
    )
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = (
        build_output_tables(data['train'])
    )

    # Store maps in a single list for convenience
    maps = [vocab_to_index, index_to_vocab, actions_to_index, 
            index_to_actions, targets_to_index, index_to_targets]

    # List of tokenized instructions, actions, and targets for each dataset
    train_dict = {"instructions": [], "actions": [], "targets": []}
    val_dict = {"instructions": [], "actions": [], "targets": []}
    data_dicts = [train_dict, val_dict]

    # Iterate through every episode in each dataset
    for dataset, data_dict in zip(data.values(), data_dicts):
        for episode in dataset:
            # Begin each instruction with a <start> token
            inst_tokens = [vocab_to_index['<start>']]
            action_indices = []
            target_indices = []
            for inst, (action, target) in episode:
                # Tokenize and store instructions
                inst = preprocess_string(inst)

                # Word level tokenization
                for word in inst.lower().split(" "):
                    word_idx = vocab_to_index.get(word, vocab_to_index['<unk>'])                    
                    inst_tokens.append(word_idx)

                    if word_idx == vocab_to_index['<unk>']:
                        unk_count += 1 
                    else:
                        word_count += 1
                
                # Store action and target indices
                action_indices.append(actions_to_index[action])
                target_indices.append(targets_to_index[target])

            # Truncate instruction tokens to max_seq_len
            if len(inst_tokens) > max_seq_len:
                inst_tokens = inst_tokens[:max_seq_len]
            if len(action_indices) > max_output_len:
                action_indices = action_indices[:max_output_len]
            if len(target_indices) > max_output_len:
                target_indices = target_indices[:max_output_len]
            
            # Add <pad> and <end> tokens where applicable
            if len(inst_tokens) < max_seq_len:
                for _ in range(max_seq_len - len(inst_tokens) - 1):
                    inst_tokens.append(vocab_to_index['<pad>'])
                inst_tokens.append(vocab_to_index['<end>'])
            
            if len(action_indices) < max_output_len:
                action_indices.append(actions_to_index['<end>'])            
                for _ in range(max_output_len - len(action_indices)):
                    action_indices.append(actions_to_index['<pad>'])

            if len(target_indices) < max_output_len:
                target_indices.append(targets_to_index['<end>'])            
                for _ in range(max_output_len - len(target_indices)):
                    target_indices.append(targets_to_index['<pad>'])

            # Store tokens in relevant data dict
            data_dict['instructions'].append(inst_tokens)
            data_dict['actions'].append(action_indices)
            data_dict['targets'].append(target_indices)  

        # Convert all token lists to Tensors of int64
        data_dict['instructions'] = torch.Tensor(data_dict['instructions']).to(torch.int64)
        data_dict['actions'] = torch.Tensor(data_dict['actions']).to(torch.int64)
        data_dict['targets'] = torch.Tensor(data_dict['targets']).to(torch.int64)

        # Perform one-hot encoding on actions and targets
        # Note that tokens do not require one-hot embedding because
        # torch.nn.Embedding() takes as an input a tensor of integers
        # # TODO: Do I actually need one hot?
        # data_dict['actions'] = F.one_hot(data_dict['actions'],
        #                                  num_classes = len(actions_to_index))
        # data_dict['targets'] = F.one_hot(data_dict['targets'],
        #                                  num_classes = len(targets_to_index))
    
    # Print some tokenization details
    tokenizer_time = round(time.time() - begin_tokenizer, 2)
    print(f'Word-level tokenizer time: {tokenizer_time} seconds')
    print(f'Percent of tokens unknown: {round(unk_count/word_count * 100, 2)}%')

    print()
    print()
    print(data_dict['instructions'][2].shape)
    print(data_dict['actions'][2].shape)
    print(data_dict['targets'][2].shape)
    print(data_dict['instructions'][3].shape)
    print(data_dict['actions'][3].shape)
    print(data_dict['targets'][3].shape)
    print()
    print()

    return train_dict, val_dict, maps

def get_device(force_cpu, status=True):
    # if not force_cpu and torch.backends.mps.is_available():
    # 	device = torch.device('mps')
    # 	if status:
    # 		print("Using MPS")
    # elif not force_cpu and torch.cuda.is_available():
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespaces with one space
    s = re.sub(r"\s+", " ", s)
    # replace digits with no space
    s = re.sub(r"\d", "", s)
    return s


def build_tokenizer_table(train, vocab_size=1000):
    word_list = []
    padded_lens = []
    episode_lens = []
    for episode in train:
        inst_count = 0
        for inst, _ in episode:
            inst = preprocess_string(inst)
            padded_len = 2  # start/end
            for word in inst.lower().split():
                if len(word) > 0:
                    word_list.append(word)
                    padded_len += 1
            padded_lens.append(padded_len)
            inst_count += 1
        episode_lens.append(inst_count)
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[
        : vocab_size - 4
    ]  # save room for <pad>, <start>, <end>, and <unk>
    vocab_to_index = {w: i + 4 for i, w in enumerate(corpus_)}
    vocab_to_index["<pad>"] = 0
    vocab_to_index["<start>"] = 1
    vocab_to_index["<end>"] = 2
    vocab_to_index["<unk>"] = 3
    index_to_vocab = {vocab_to_index[w]: w for w in vocab_to_index}
    max_inst_len = int((np.average(padded_lens) + np.std(padded_lens) * 2) * \
                       (np.average(episode_lens) + np.std(episode_lens) * 2) + 0.5)
    max_output_len = int((np.average(episode_lens) + np.std(episode_lens) * 2) + 0.5)
    return (
        vocab_to_index,
        index_to_vocab,
        max_inst_len,
        max_output_len
    )


def build_output_tables(train):
    actions = set()
    targets = set()
    for episode in train:
        for _, outseq in episode:
            a, t = outseq
            actions.add(a)
            targets.add(t)
    actions_to_index = {a: i + 4 for i, a in enumerate(actions)}
    targets_to_index = {t: i + 4 for i, t in enumerate(targets)}
    actions_to_index['<pad>'] = 0
    targets_to_index['<pad>'] = 0
    actions_to_index["<end>"] = 1
    targets_to_index["<end>"] = 1
    actions_to_index["<bos>"] = 2
    targets_to_index["<bos>"] = 2
    actions_to_index["<cls>"] = 3
    targets_to_index["<cls>"] = 3
    index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
    index_to_targets = {targets_to_index[t]: t for t in targets_to_index}
    return actions_to_index, index_to_actions, targets_to_index, index_to_targets

def prefix_match(predicted_labels, gt_labels):
    # predicted and gt are sequences of (action, target) labels, the sequences should be of same length
    # computes how many matching (action, target) labels there are between predicted and gt
    # is a number between 0 and 1 

    seq_length = len(gt_labels)
    
    for i in range(seq_length):
        if predicted_labels[i] != gt_labels[i]:
            break
    
    pm = (1.0 / seq_length) * i

    return pm