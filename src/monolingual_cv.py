#!/usr/bin/python3

from __future__ import print_function
import numpy as np
np.random.seed(1337)

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Activation, Input, merge
from keras.layers import Embedding, Convolution1D, MaxPooling1D
from keras.layers import AveragePooling1D, LSTM, GRU
from keras.utils import np_utils
import sys, time, re, glob
from collections import defaultdict
from gensim.utils import simple_preprocess

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

_white_spaces = re.compile(r"\s\s+")

maxchars = 200
embedding_dims = 100
batch_size = 32
nb_epoch = 10
minfreq = 0
data_path = sys.argv[1]
minwordfreq = 15
maxwordlen = 400
seed = 1234

import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def read_data():
    """
    Reads all files in sys.argv[1]
    """
    labels = []
    documents = []
    for data_file in glob.iglob(sys.argv[1]+"/*"): # iglob vs glob : iterator, just require less memory
        doc = open(data_file, "r").read().strip()
        wrds = doc.split(" ")
        label = data_file.split("/")[-1].split(".txt")[0].split("_")[-1] # Get CEFR score from file name
        if label == "EMPTY": continue
        if len(wrds) >= maxwordlen:
            doc = " ".join(wrds[:maxwordlen]) # troncate text if more than 400 words
        doc = _white_spaces.sub(" ", doc) # replace any white space sequence to a single space
        labels.append(label)
        documents.append(doc)

    return (documents, labels)

def char_tokenizer(s):
    return list(s)

def word_tokenizer(s):
    """
    Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long. (default 2 to 15)
    Uses tokenize internally
    """
    return simple_preprocess(s)

def getWords(D):
    """
    Input :
    D = [
        "A sentence limited to 400 words (space separated)",
        ...
    ]

    Output:
    wordSet = {
        token1 (str): nb_occ (int),
        token2 (str): nb_occ (int),
        ...
    }
    max_features: nb (int) # nb tokens with occurrence > minwordfreq (15)
    """
    wordSet = defaultdict(int)
    max_features = 3
    # Count token occurences
    for d in D:
        for c in word_tokenizer(d):
            wordSet[c] += 1
    for c in wordSet:
        if wordSet[c] > minwordfreq:
            max_features += 1
    return wordSet, max_features

def getChars(D):
    charSet = defaultdict(int)
    max_features = 3
    for d in D:
        for c in char_tokenizer(d):
            charSet[c] += 1
    for c in charSet:
        if charSet[c] > minfreq:
            max_features += 1
    return charSet, max_features

def transform(D, vocab, minfreq, tokenizer="char"):
    """
    Encode input documents.

    Input:
    D = [
        "A sentence limited to 400 words (space separated)",
        ...
    ]
    vocab = {
        token1 (str): nb_occ (int),
        token2 (str): nb_occ (int),
        ...
    }
    minfreq: (int)
    """
    # Attribute an index (count) to each token in vocab with nb_occ > minfreq
    features = defaultdict(int)
    count = 0
    for i, k in enumerate(vocab.keys()):
        if vocab[k] > minfreq:
            features[k] = count
            count += 1

    start_char = 1
    oov_char = 2
    index_from = 3

    X = []
    # Tokenize docs by word or char
    for j, d in enumerate(D):
        x = [start_char] # add a start_char for each doc, with ID 1
        z = None
        if tokenizer == "word":
            z = word_tokenizer(d)
        else:
            z = char_tokenizer(d)

        # For each token, get nb_occ
        for c in z:
            freq = vocab[c]
            if c in vocab:
                # Tokens in features are the one with sufficient frequency
                if c in features:
                    x.append(features[c]+index_from)
                else:
                    x.append(oov_char)
            else:
                continue
        X.append(x)
    return X

################################################################################
################################################################################

# Reading training set
print("Reading the training set... ", end="")
sys.stdout.flush()
pt = time.time()
doc_train, y_labels = read_data()
print(time.time() - pt)

# Get vocab & encode training set
print("Transforming the datasets... ", end="")
sys.stdout.flush()
pt = time.time()
word_vocab, max_word_features = getWords(doc_train)
print("Number of features= ", max_word_features)
x_word_train = transform(doc_train, word_vocab, minwordfreq, tokenizer="word")
print(len(x_word_train), 'train sequences') # 1033
print(time.time() - pt)

# Pad sequences with keras built-in function
# By default adds 0s at the BEGINNING of sequences
print('Pad sequences (samples x time)')
x_word_train = sequence.pad_sequences(x_word_train, maxlen=maxwordlen)

# Encode labels
print("Transforming the labels... ", end="")
sys.stdout.flush()
pt = time.time()
unique_labels = list(set(y_labels))
print("Class labels = ",unique_labels)
n_classes = len(unique_labels)
indim = x_word_train.shape[1] # 400 here
y_labels = [unique_labels.index(y) for y in y_labels]

# to_categorical: Converts a class vector (integers) to binary class matrix.
y_train = np_utils.to_categorical(np.array(y_labels), len(unique_labels))
print('y_train shape:', y_train.shape) # 6 classes
print(time.time() - pt)

cv_accs, cv_f1 = [], [] # cv_accs not used
k_fold = StratifiedKFold(10, random_state=seed)
all_gold = []
all_preds = []
i = 1
for train, test in k_fold.split(x_word_train, y_labels):

    pt = time.time()
    print(f'Build model num {i}...')

    model = Sequential()

    # Embedding: Turns positive integers (indexes) into dense vectors of fixed size.
    # max_word_features = input_dim = nb of different values taken by the integers = 804 here
    # embedding_dims = output_dim = 100 here
    # maxwordlen = input_length = size of input elements (one doc) 400 here
    model.add(Embedding(max_word_features, embedding_dims, input_length=maxwordlen))

    #model.add(GRU(50))
    #model.add(AveragePooling1D(pool_length=8))

    # Flatten: Flattens the input. Does not affect the batch size.
    model.add(Flatten())

    # Dense: Just your regular densely-connected NN layer.
    # n_classes = units = nb out neurons
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])

    model.fit(x_word_train[train], y_train[train],
              batch_size=batch_size,
              epochs=nb_epoch)

    predict_x=model.predict(x_word_train[test])
    y_pred=np.argmax(predict_x,axis=1)

    pred_labels = [unique_labels[x] for x in y_pred]
    gold_labels = [unique_labels[x] for x in np.array(y_labels)[test]]
    all_gold.extend(gold_labels)
    all_preds.extend(pred_labels)

    cv_f1.append(f1_score(np.array(y_labels)[test], y_pred, average="weighted"))
    print()

    i += 1

print(f"\nAll F1-scores of the {i} models:", cv_f1,sep="\n")
print("Average F1 scores", np.mean(cv_f1))
print(confusion_matrix(all_gold,all_preds))
