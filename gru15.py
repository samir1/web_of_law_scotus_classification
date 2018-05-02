from __future__ import print_function

import argparse
import os
import sys
import numpy as np
from textacy.datasets.supreme_court import SupremeCourt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input
from keras.layers import Embedding, CuDNNGRU
from keras.layers import Dropout
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import h5py
from tensorflow.python.lib.io import file_io
from time import gmtime, strftime
import pickle


def train_model():

    if not os.path.exists('ModelCheckpoint'):
        os.makedirs('ModelCheckpoint')

    MAX_SEQUENCE_LENGTH = 90018
    MAX_NB_WORDS = 170000
    EMBEDDING_DIM = 300
    VALIDATION_SPLIT = 0.1
    BATCH_SIZE = 32


    print('Indexing word vectors.')

    embeddings_index = {}
    f = file_io.FileIO('GoogleNews-vectors-negative300.txt', mode='r')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))


    print('Processing text dataset')

    sc = SupremeCourt()
    print(sc.info)

    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids

    issue_codes = list(sc.issue_area_codes.keys()) # 15 labels
    issue_codes.sort()
    issue_codes = [str(ic) for ic in issue_codes]

    labels_index = dict(zip(issue_codes, np.arange(len(issue_codes))))

    for record in sc.records():
        if record['issue'] == None: # some cases have None as an issue
            labels.append(labels_index['-1'])
        else:
            labels.append(labels_index[record['issue'][:-4]])
        texts.append(record['text'])

    print('Found %s texts.' % len(texts))
    print('Found %s labels.' % len(labels_index))

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences)

    MAX_SEQUENCE_LENGTH = data.shape[1]

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=VALIDATION_SPLIT, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=VALIDATION_SPLIT, random_state=42)


    def generator():
        while True:
            indices = list(range(len(x_train)))
            imax = len(indices)//BATCH_SIZE
            for i in range(imax):
                list_IDs_temp = indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                yield x_train[list_IDs_temp], y_train[list_IDs_temp]

    def test_generator():
        while True:
            indices = list(range(len(x_test)))
            imax = len(indices)//BATCH_SIZE
            for i in range(imax):
                list_IDs_temp = indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                yield x_test[list_IDs_temp], y_test[list_IDs_temp]

    def val_generator():
        while True:
            indices = list(range(len(x_val)))
            imax = len(indices)//BATCH_SIZE
            for i in range(imax):
                list_IDs_temp = indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                yield x_val[list_IDs_temp], y_val[list_IDs_temp]


    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


    print('Training model.')

    model = Sequential()
    model.add(
      Embedding(num_words,
                EMBEDDING_DIM,
                weights=[embedding_matrix],
                input_length=MAX_SEQUENCE_LENGTH,
                trainable=False)
    )
    model.add(Dropout(0.25))
    model.add(CuDNNGRU(128))
    model.add(Dropout(0.5))
    model.add(Dense(len(labels_index), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    checkpointer = ModelCheckpoint(filepath="ModelCheckpoint/" + os.path.basename(__file__)[:-3] +
        "-{epoch:02d}-{val_acc:.2f}.hdf5",
                                   monitor='val_acc',
                                   verbose=2,
                                   save_best_only=True,
                                   mode='max')

    earlystopper = EarlyStopping(monitor='val_loss',
                             min_delta=0,
                             patience=0,
                             verbose=2,
                             mode='auto')

    model.summary()

    model.fit_generator(generator=generator(),
                        steps_per_epoch = len(x_train)//BATCH_SIZE,
                        epochs=50,
                        verbose=2,
                        validation_data=test_generator(),
                        validation_steps=len(x_test)//BATCH_SIZE,
                        callbacks=[checkpointer, earlystopper],
                        shuffle=True)

    score = model.evaluate_generator(val_generator(),
                                     steps=len(x_val)//BATCH_SIZE)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Save Keras ModelCheckpoints locally
    model.save('model.hdf5')

if __name__ == '__main__':
    train_model()
