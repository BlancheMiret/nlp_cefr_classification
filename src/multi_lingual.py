#!/usr/bin/python3

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Activation, Input, concatenate
from keras.layers import Embedding, Convolution1D, MaxPooling1D
from keras.layers import AveragePooling1D, LSTM, GRU
from keras.utils import np_utils
import sys, time, re, glob
from collections import defaultdict
from gensim.utils import simple_preprocess
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from keras.layers import SpatialDropout1D

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

seed = 1234
_white_spaces = re.compile(r"\s\s+")
maxlen = 2000
embedding_dims = 16
batch_size = 128
nb_epoch = 8
nb_filter = 50
filter_length = 5
pool_length = 5
minwordfreq = 15
mincharfreq = 0
maxwordlen = 400

tokenizer_re = re.compile("\w+|\S")

def read_data():
    """
    Reads all file, made to get all plain file from the 3 languages.
    """
    labels = []
    lg_labels = []
    documents = []
    for data_file in glob.iglob(sys.argv[1]+"/*/*"):
        lang_label = data_file.split("/")[-2]

        if "RemovedFiles" in data_file: continue
        if "parsed" in data_file: continue
        if "MERLIN" in data_file: continue
        if "features" in data_file: continue
        doc = open(data_file, "r").read().strip()
        wrds = doc.split(" ")

        label = data_file.split("/")[-1].split(".txt")[0].split("_")[-1]
        if label in ["EMPTY", "unrated"]: continue
        if len(wrds) >= maxwordlen:
            doc = " ".join(wrds[:maxwordlen])
        doc = _white_spaces.sub(" ", doc)
        labels.append(label)
        lg_labels.append(lang_label)
        documents.append(doc)

    return (documents, labels, lg_labels)

def char_tokenizer(s):
    return list(s)

def word_tokenizer(s):
    return tokenizer_re.findall(s)

def getWords(D):
    wordSet = defaultdict(int)
    max_features = 3
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
        if charSet[c] > mincharfreq:
            max_features += 1
    return charSet, max_features

def transform(D, vocab, minfreq, tokenizer="char"):
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
    for j, d in enumerate(D):
        x = [start_char]
        z = None
        if tokenizer == "word":
            z = word_tokenizer(d)
        else:
            z = char_tokenizer(d)
        for c in z:
            freq = vocab[c]
            if c in vocab:
                if c in features:
                    x.append(features[c]+index_from)
                else:
                    x.append(oov_char)
            else:
                continue
        X.append(x)
    return X

# Read all plain files from 3 languages
print("Reading the training set... ", end="")
sys.stdout.flush()
doc_train, y_labels, y_lang_labels = read_data()

# Get vocab & encode training set
# at char & word levels
print("Transforming the datasets... ", end="")
sys.stdout.flush()
word_vocab, max_word_features = getWords(doc_train)
char_vocab, max_char_features = getChars(doc_train)
print("Number of word features= ", max_word_features, " char features= ", max_char_features) # 2245 & 159
x_char_train = transform(doc_train, char_vocab, mincharfreq, tokenizer="char")
x_word_train = transform(doc_train, word_vocab, minwordfreq, tokenizer="word")
print(len(x_char_train), 'train sequences') # 2 267

# Pad sequences with keras built-in function
# By default adds 0s at the BEGINNING of sequences
x_char_train = sequence.pad_sequences(x_char_train, maxlen=maxlen) #2000
x_word_train = sequence.pad_sequences(x_word_train, maxlen=maxwordlen) # 400

# Encode labels
unique_labels = list(set(y_labels))
lang_labels = ["CZ", "IT", "DE"]
print("Class labels = ",unique_labels)
n_classes = len(unique_labels) # 6
n_grp_classes = len(lang_labels) # 3
grp_train, grp_test = [], []

y_labels = [unique_labels.index(y) for y in y_labels] # list 2267 of value 0-5
y_lang_labels = [lang_labels.index(y) for y in y_lang_labels] # list 2267 of value 0-2

y_train = np_utils.to_categorical(np.array(y_labels), len(unique_labels))
lng_train = np_utils.to_categorical(np.array(y_lang_labels), len(lang_labels))

cv_accs, cv_f1 = [], []
k_fold = StratifiedKFold(10,random_state=seed, shuffle=True)
n_iter = 1
all_golds = []
all_preds = []

for train, test in k_fold.split(x_word_train, y_labels):
    print('Build model... ', n_iter)
    char_input = Input(shape=(maxlen, ), dtype='int32', name='char_input')
    charx = Embedding(max_char_features, 16, input_length=maxlen)(char_input)
    charx = SpatialDropout1D(0.25)(charx)

    word_input = Input(shape=(maxwordlen, ), dtype='int32', name='word_input')
    wordx = Embedding(max_word_features, 32, input_length=maxwordlen)(word_input)
    wordx = SpatialDropout1D(0.25)(wordx)

    charx = Flatten()(charx)
    wordx = Flatten()(wordx)

    y = concatenate([charx, wordx])

    # predict language first
    grp_predictions = Dense(n_grp_classes, activation='softmax')(y)

    # then predict class after adding language info to input tensor
    y = concatenate([y, grp_predictions])
    y = Dropout(0.25)(y)
    y_predictions = Dense(n_classes, activation='softmax')(y)

    model = Model(inputs=[char_input, word_input], outputs=[y_predictions, grp_predictions])

    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'], loss_weights=[1.0, 0.5])

    hist = model.fit([x_char_train[train], x_word_train[train]], [y_train[train], lng_train[train]],
              batch_size=batch_size,
              epochs=nb_epoch)

    y_pred = model.predict([x_char_train[test], x_word_train[test]])
    y_classes = np.argmax(y_pred[0], axis=1)
    y_gold = np.array(y_labels)[test]

    # Transform id predict to class str
    pred_labels = [unique_labels[x] for x in y_classes]
    gold_labels = [unique_labels[x] for x in y_gold]
    all_golds.extend(gold_labels)
    all_preds.extend(pred_labels)
    cv_f1.append(f1_score(y_gold, y_classes, average="weighted"))
    n_iter += 1
    print()

print(f"\nAll F1-scores for {n_iter} models:", cv_f1,sep="\n")
print("Average F1 scores", np.mean(cv_f1))
print(confusion_matrix(all_golds,all_preds))
