import os
import re
import json
import gensim
import tqdm
import numpy as np
from collections import Counter
from spacy.lang.en import English
import torch


def process_book_dir(d, max_per_book=None):
    nlp = English()
    nlp.add_pipe("sentencizer")
    nlp.max_length = 3293518  # we aren't doing any heavy parsing or anything; set based on biggest book
    processed_text_lines = []
    n_files = 0
    for root, dirs, fns in os.walk(d):
        for fn in fns:
            if fn.split(".")[-1] == "txt":
                n_files += 1
                book_title = fn.split(".")[0]
                with open(os.path.join(d, fn), "r") as f:
                    new_lines = [s for s in f.readlines()]
                    if max_per_book is not None and max_per_book < len(new_lines):
                        new_lines = new_lines[:max_per_book]
                    entire_book = " ".join(new_lines)
                    doc = nlp(entire_book)
                    sentences = list(doc.sents)
                    sentences = [preprocess_string(str(s)) for s in sentences]
                    processed_text_lines.extend(
                        [[s, book_title] for s in sentences if len(s) > 0]
                    )
    processed_text_lines = [
        [line, label] for (line, label) in processed_text_lines if len(line.split()) > 0
    ]
    # processed_text_lines = processed_text_lines[:int((len(processed_text_lines) / 1000))]
    print(
        "read in %d lines from %d files in directory %s"
        % (len(processed_text_lines), n_files, d)
    )
    return processed_text_lines


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespaces with one space
    s = re.sub(r"\s+", " ", s)
    # replace digits with no space
    s = re.sub(r"\d", "", s)
    # Lowercase string
    s = s.lower()
    return s


def build_tokenizer_table(train, vocab_size):
    word_list = []
    padded_lens = []
    inst_count = 0
    counts_per_book = {}
    for line, book in train:
        line = preprocess_string(line)
        padded_len = 2  # start/end
        if book not in counts_per_book:
            counts_per_book[book] = Counter()
        for word in line.lower().split():
            if len(word) > 0:
                word_list.append(word)
                padded_len += 1
                counts_per_book[book][word] += 1
        padded_lens.append(padded_len)
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
    return (
        vocab_to_index,
        index_to_vocab,
        int(np.max(padded_lens)),  # we don't need a cutoff for vanilla LM
    )

def read_analogies(analogies_fn):
    with open(analogies_fn, "r") as f:
        pairs = json.load(f)
    return pairs


def save_word2vec_format(fname, model, i2v):
    print("Saving word vectors to file...")  # DEBUG
    with gensim.utils.smart_open(fname, "wb") as fout:
        fout.write(
            gensim.utils.to_utf8("%d %d\n" % (model.vocab_size, model.embedding_dim))
        )
        # store in sorted order: most frequent words at the top
        for index in tqdm.tqdm(range(len(i2v))):
            word = i2v[index]
            row = model.embed.weight.data[index]
            fout.write(
                gensim.utils.to_utf8(
                    "%s %s\n" % (word, " ".join("%f" % val for val in row))
                )
            )


def encode_data(data, v2i, seq_len):
    num_insts = sum([len(ep) for ep in data])
    x = np.zeros((num_insts, seq_len), dtype=np.int32)
    lens = np.zeros((num_insts, 1), dtype=np.int32)

    idx = 0
    n_early_cutoff = 0
    n_unks = 0
    n_tks = 0
    for sent, source in data:
        x[idx][0] = v2i["<start>"]
        jdx = 1
        for word in sent.split():
            if len(word) > 0:
                x[idx][jdx] = v2i[word] if word in v2i else v2i["<unk>"]
                n_unks += 1 if x[idx][jdx] == v2i["<unk>"] else 0
                n_tks += 1
                jdx += 1
                if jdx == seq_len - 1:
                    if len(sent.split()) >= seq_len:
                        n_early_cutoff += 1
                    break
        x[idx][jdx] = v2i["<end>"]
        lens[idx][0] = jdx
        idx += 1
    print(
        "INFO: had to represent %d/%d (%.4f) tokens as unk with vocab limit %d"
        % (n_unks, n_tks, n_unks / n_tks, len(v2i))
    )
    print(
        "INFO: cut off %d sentences at len %d before true ending"
        % (n_early_cutoff, seq_len)
    )
    print("INFO: encoded %d sentences without regard to order" % idx)

    return x, lens

def get_tokens_and_labels(encoded_sentences, lens, args):
    """
    Given ndarrays of encoded sentences and sentence lenghts,
    returns two ndarrays:
    - tokens: ndarray of token indices
    - labels: ndarray of multihot tensor labels of context tokens
    
    labels is of size (# of tokens, size of vocab) and provides 
    multihot encoding where all tokens in the context are labelled 1. 
    """
    # Initialize tokens and labels as empty lists
    tokens, labels = [], []

    tokens = torch.zeros((int(sum(lens))))
    labels = torch.zeros((int(sum(lens)), args.vocab_size))

    # For each token in each sentence
    row = 0
    for sentence, slen in zip(encoded_sentences, lens):
        for token_idx in range(slen[0]):
            # Append the token to tokens
            tokens[row] = sentence[token_idx]

            # Append the token's context to labels as a multihot tensor
            for label_idx in range(token_idx - args.context_size, 
                                   token_idx + args.context_size + 1):
                # Don't include the token itself in the context
                if label_idx == token_idx:
                    continue
                # Handle padding
                elif label_idx < 0 or label_idx >= slen:
                    labels[row, 0] = 1
                # Flip each context token's label to 1
                else:
                    labels[row, sentence[label_idx]] = 1
            # Increment row of tokens and labels
            row += 1
    
    return tokens, labels