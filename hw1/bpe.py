import re
import json
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from utils import build_output_tables, preprocess_string
from tqdm import tqdm
import time

def get_best_byte_pair(vocab) -> tuple:
    """
    Given a vocab dictionary of words broken up into tokens mapped
    to their frequency, returns the most frequent pair of tokens.
    """
    pair_freq = dict()
    # Iterate through each word in the vocabulary
    for word, freq in vocab.items():
        # Look at the individual tokens in each word
        tokens = word.split()
        for i in range(len(tokens) - 1):
            # Pair consecutive tokens and count their frequency
            token_pair = (tokens[i], tokens[i+1])
            if token_pair not in pair_freq.keys():
                pair_freq[token_pair] = 0
            pair_freq[token_pair] += freq
    
    # Return the most frequent paired token
    best_pair = max(pair_freq, key=pair_freq.get)
    return best_pair

def merge_vocab(v_in, bp: tuple):
    """
    Given a vocab dictionary of words split into subwords and
    their frequencies, replaces subwords with the given byte pair
    and returns the updated frequencies. 
    """
    # Create a regex object of the two tokens to be replaced
    bigram = re.escape(' '.join(bp))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    v_out = dict()
    for word, freq in v_in.items():
        # Replace the bigram with the new byte-pair and store
        w_out = p.sub(''.join(bp), word)
        v_out[w_out] = freq
    return v_out

def get_tokens(vocab):
    """
    From the dictionary of words separated into byte pairs,
    return a set of the most common byte pairs. 
    """
    tokens = set()
    for word, freq in vocab.items():
        subwords = word.split()
        for i in subwords:
            tokens.add(i)
    return tokens

def build_bpe_table(train, vocab_size=1000):
    """
    Implements BPE as described in Sennrich, Haddow, and Birch 2016. 
    They provide helpful psuedocode, and I also used this tutorial:
    https://leimao.github.io/blog/Byte-Pair-Encoding/
    """
    # Count the occurences of each word in the training data
    subword_list = []
    for episode in train:
        for inst, _ in episode:
            inst = preprocess_string(inst)
            for word in inst.lower().split():
                if len(word) > 0:
                    # Separate words into characters
                    subword = " ".join(word) 

                    # Append an end-of-word token
                    subword = subword+ " </w>"
                    subword_list.append(subword)
    
    # Build counter dictionaries of subwords
    vocab = Counter(subword_list)

    # Initialize the set of tokens
    token_set = set(("<pad>", "<start>", "<end>", "<unk>"))

    # Merge the most frequent character pair at each iteration
    merge_num = 1
    while True:
        byte_pair = get_best_byte_pair(vocab)
        vocab = merge_vocab(vocab, byte_pair)

        # Add any additional tokens to your token set
        new_tokens = get_tokens(vocab)
        for i in new_tokens:
            token_set.add(i)

        # Always verbose for BPE
        if True:
            print('Merge #{}'.format(merge_num))
            print('Most common pair: {}'.format(byte_pair))
            print('Number of tokens: {}'.format(len(token_set)))
            print('==========')
            merge_num += 1
        
        # Stop merging once we have 1000 tokens
        if len(token_set) >= 1000:
            break
    
    # Sort tokens by character length
    # To encode in BPE, we need to start with the longest tokens
    token_lens = dict()
    for t in token_set:
        # Shorter </w> to a single character and store the length
        t_adjusted = t.replace("</w>", "$")
        token_lens[t] = len(t_adjusted)

    # Perform the sort by token length
    token_lens = dict(sorted(token_lens.items(), 
                        key=lambda item: item[1], 
                        reverse=True))

    # Store only the tokens themselves     
    token_list = list(token_lens.keys())
    
    # Generate the output dictionaries mapping tokens to indices
    tokens_to_index = {t: i for i, t in enumerate(token_list)}
    index_to_tokens = {i: t for t, i in tokens_to_index.items()}

    # Print entire BPE dictionary
    for k, v in tokens_to_index.items():
        print(f"{k} - {v}")
    
    return tokens_to_index, index_to_tokens

def tokenize_bpe(args):
    """
    Uses Byte-Pair Encoding to tokenize the entire instruciton set. 
    Returns the tokenized training data, validation data, and maps. 

    Original implementation based on Sennrich, Haddow, and Birch, 2016
    Link to paper here: https://arxiv.org/pdf/1508.07909.pdf
    Runs successfully in about 10 minutes -- how do I speed this up?
    """
    # Code for Tokenizer Analysis
    begin_tokenizer = time.time()

    # Load the data from the json
    data = json.load(open(args.in_data_fn))

    # Use the utils to construct the necessary maps
    tokens_to_index, index_to_tokens = (
        build_bpe_table(data['train'], vocab_size = args.vocab_size)
    )
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = (
        build_output_tables(data['train'])
    )

    # Store maps in a single list for convenience
    maps = [tokens_to_index, index_to_tokens, actions_to_index, 
            index_to_actions, targets_to_index, index_to_targets]

    # List of tokenized instructions, actions, and targets for each dataset
    train_dict = {"instructions": [], "actions": [], "targets": []}
    val_dict = {"instructions": [], "actions": [], "targets": []}
    data_dicts = [train_dict, val_dict]

    # Track statistics to determine max_seq_len
    token_lens = []
    in_training = False
    
    # Iterate through every episode in each dataset
    for dataset, data_dict in zip(data.values(), data_dicts):
        in_training = not in_training
        for episode in tqdm(dataset):
            for inst, outseq in episode:
                # Begin each instruction with a <start> token
                inst_tokens = [tokens_to_index['<start>']]
                inst = preprocess_string(inst)

                # Tokenize each word of the instruction individually
                for word in inst.lower().split(" "):
                    word_tokens = encode_bpe_word(word, tokens_to_index)
                    inst_tokens.extend(word_tokens)

                # Tokenize and store action and target
                a, t = outseq

                # Store tokens in relevant data dict
                data_dict['instructions'].append(inst_tokens)
                data_dict['actions'].append(actions_to_index[a])
                data_dict['targets'].append(targets_to_index[t])  

                # Store statistics for calculating max_seq_len
                if in_training:
                    # Add 1 for end token
                    token_lens.append(len(inst_tokens) + 1)

    # Calculate the maximum sequence length
    max_seq_len = int(np.average(token_lens) + 2 * np.std(token_lens) + 0.5)
    for data_dict in data_dicts:
        # Apply max_seq_len to each tokenized list of instructions
        for i, token_list in enumerate(data_dict['instructions']):
            # Truncate instruction tokens to max_seq_len
            if len(token_list) > max_seq_len:
                data_dict['instructions'][i] = token_list[:max_seq_len]
            
            # Add <pad> and <end> tokens where applicable
            if len(token_list) < max_seq_len:
                for _ in range(int(max_seq_len - len(data_dict['instructions'][i]) - 1)):
                    data_dict['instructions'][i].append(tokens_to_index['<pad>'])
                data_dict['instructions'][i].append(tokens_to_index['<end>'])

            assert len(data_dict['instructions'][i]) == max_seq_len

        # Convert all token lists to Tensors of int64
        data_dict['instructions'] = torch.Tensor(data_dict['instructions']).to(torch.int64)
        data_dict['actions'] = torch.Tensor(data_dict['actions']).to(torch.int64)
        data_dict['targets'] = torch.Tensor(data_dict['targets']).to(torch.int64)

        # Perform one-hot encoding on actions and targets
        # Note that tokens do not require one-hot embedding because
        # torch.nn.Embedding() takes as an input a tensor of integers
        data_dict['actions'] = F.one_hot(data_dict['actions'],
                                         num_classes = len(actions_to_index))
        data_dict['targets'] = F.one_hot(data_dict['targets'],
                                         num_classes = len(targets_to_index))
    
    print(f'BPE Tokenizer Time: {time.time() - begin_tokenizer}')
    print("No Unknown Tokens")

    return train_dict, val_dict, maps

def encode_bpe_word(string, tokens_to_index):
    """
    Custom function for encoding a single word in BPE. 
    Returns a list of indices of BPE tokens. 

    This is much simpler than the online version I referenced!
    tokenize_word() from https://leimao.github.io/blog/Byte-Pair-Encoding/ 
    I wonder if I'm missing any edge cases -- manual inspection looks good
    but I don't have any proper testing yet. It doesn't handle unknown tokens. 
    """
    output_tokens = []
    string = string + "</w>"
    # Loop through tokens from longest to shortest
    for token, idx in tokens_to_index.items():
        # Find the last occurence of the token in the string
        last = string.rfind(token)

        # If the token appears in the string
        while last != -1:
            # Replace it with a space
            string = string[:last] + " " + string[last + len(token):]

            # The number of previous tokens == the number of previous spaces
            previous_tokens = string[:last].count(' ')

            # Insert the idx of this token after all of the previous tokens
            output_tokens.insert(previous_tokens, idx)

            # Check for another instance of the token and potentially repeat
            last = string.rfind(token)
        
        if string.replace(" ", "") == "":
            break

    # TODO: Store any unknown tokens. Cool but not necessary given our preprocessing. 

    return output_tokens