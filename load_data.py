# Import libraries
from nltk.stem import WordNetLemmatizer
from random import shuffle
import pandas as pd
import numpy as np
import datasets
import nltk
import re

nltk.download('wordnet')
nltk.download('omw-1.4')

root = WordNetLemmatizer()


def clean_word(token):
    token = token.lower()                    # lower words
    token = re.sub("[^a-z0-9]*", "", token)  # remove redundant characters
    token = root.lemmatize(token)            # lemmatize words

    return token


def tokenize_sentences(sent, max_length, token2id, unk=1, pad=0):
    tokens = [token2id.get(clean_word(token), unk) for token in sent.split()]

    if len(tokens) < max_length:
        diff = max_length - len(tokens)
        tokens.extend([pad] * diff)
    elif len(tokens) > max_length:
        tokens = tokens[:max_length]

    return tokens


def load_glove(glove_path):
    token2idx = {'<PAD>': 0, '<UNK>': 1}
    counter = 2
    # Open GLoVe file for word embedding
    with open(glove_path, 'r') as f:
        raw_glove = f.read().strip().split('\n')

    # Create an empty array to fill with glive embeddings
    glove_weights = np.zeros((len(raw_glove), len(raw_glove[0].split()) - 1), dtype=float)

    # Store values of embeddings and create dictionary with words and tokens
    for idx, item in enumerate(raw_glove):
        for idy, entity in enumerate(item.split()):
            if idy == 0:
                token2idx[entity] = counter
                counter += 1
            else:
                glove_weights[idx, idy - 1] = float(entity)

    return token2idx, glove_weights


def load_dataset(dataset_name, data_len, glove_path, max_len, test_size):
    # Create variables
    dataset = []
    s1, s2 = [], []

    # Load dataset
    data = datasets.load_dataset(dataset_name, split='train')

    # Split data to a fixed size
    sentences = data['questions'][:data_len]
    labels = data['is_duplicate'][:data_len]

    # Iterate over sentences to split sentence 1 and 2
    for item in sentences:
        s1.append(item['text'][0])
        s2.append(item['text'][1])

    token2idx, glove_weights = load_glove(glove_path)

    # Create a dataset with tokens and labels
    for idx, (sent1, sent2) in enumerate(zip(s1, s2)):
        tag = 1 if labels[idx] == True else 0
        dataset.append((tokenize_sentences(sent1, max_len, token2idx),
                        tokenize_sentences(sent2, max_len, token2idx),
                        tag))

    # Shuffle dataset and split train and test
    shuffle(dataset)
    test_index = int(len(dataset) * test_size)
    train = dataset[test_index:]
    test = dataset[:test_index]

    # Create x and y of train and test from dataset
    x_train1 = np.zeros((len(train), max_len), dtype=int)
    x_train2 = np.zeros((len(train), max_len), dtype=int)
    y_train = np.zeros((len(train)), dtype=int)
    x_test1 = np.zeros((len(test), max_len), dtype=int)
    x_test2 = np.zeros((len(test), max_len), dtype=int)
    y_test = np.zeros((len(test)), dtype=int)

    for idx, item in enumerate(train):
        x_train1[idx, :] = item[0]
        x_train2[idx, :] = item[1]
        y_train[idx] = item[2]

        if idx <= len(test) - 1:
            x_test1[idx, :] = test[idx][0]
            x_test2[idx, :] = test[idx][1]
            y_test[idx] = test[idx][2]

    return (x_train1, x_train2, y_train), (x_test1, x_test2, y_test), glove_weights
