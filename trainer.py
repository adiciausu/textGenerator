from __future__ import print_function

import os

from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional
from keras.layers import LSTM
import numpy as np
import random
import sys
import io
from keras.utils.data_utils import get_file

class Trainer:
    SEQUENCE_LEN = 10
    STEP = 1
    text = None
    model = None
    words = None
    unique_words = None

    def run(self):
        if not os.path.exists('data'):
            os.mkdir('data')
        if not os.path.exists(os.path.join('data', 'logs')):
            os.mkdir(os.path.join('data', 'logs'))
        if not os.path.exists(os.path.join('data', 'checkpoints')):
            os.mkdir(os.path.join('data', 'checkpoints'))

        path = get_file(
            'nietzsche.txt',
            origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        with io.open(path, encoding='utf-8') as f:
            self.text = f.read().lower()

        self.words = [w for w in self.text.split(' ') if w.strip() != '' or w == '\n']
        print('Corpus length in letters:', len(self.text))
        print('Corpus length in words:', len(self.words))

        self.unique_words = sorted(set(self.words))
        print('Unique words:', len(self.unique_words))

        self.word_indices = dict((c, i) for i, c in enumerate(self.unique_words))
        self.indices_word = dict((i, c) for i, c in enumerate(self.unique_words))

        sequences = []
        next_words = []
        for i in range(0, len(self.words) - self.SEQUENCE_LEN, self.STEP):
            sequences.append(self.words[i: i + self.SEQUENCE_LEN])
            next_words.append(self.words[i + self.SEQUENCE_LEN])
        print('sentences count:', len(sequences))

        print('Vectorization...')
        x = np.zeros((len(sequences), self.SEQUENCE_LEN, len(self.unique_words)), dtype=np.bool)
        y = np.zeros((len(sequences), len(self.unique_words)), dtype=np.bool)
        for i, sequence in enumerate(sequences):
            for t, word in enumerate(sequence):
                x[i, t, self.word_indices[word]] = 1
            y[i, self.word_indices[next_words[i]]] = 1

        self.model = self.build_model()
        self.model.fit(x, y, validation_split=0.05, batch_size=128, epochs=60, callbacks=self.build_callbacks())

    def build_callbacks(self):
        checkpoint_path = "./data/checkpoints/text-generator-epoch{epoch:03d}-sequence%d-" \
                          "loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}" % self.SEQUENCE_LEN
        checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_acc', save_best_only=True)
        print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)
        early_stopping = EarlyStopping(monitor='val_acc', patience=10)
        tb_callback = TensorBoard(os.path.join('data', 'logs'))

        return [print_callback, checkpoint_callback, early_stopping, tb_callback]

    def build_model(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(128), input_shape=(self.SEQUENCE_LEN, len(self.unique_words))))
        model.add(Dropout(0.2))
        model.add(Dense(len(self.unique_words), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def sample(self, preds):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)

        return np.argmax(probas)

    def on_epoch_end(self, epoch, _):
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(self.text) - self.SEQUENCE_LEN - 1)

        generated = ''
        sentence = self.text[start_index: start_index + self.SEQUENCE_LEN]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        for i in range(400):
            x_pred = np.zeros((1, self.SEQUENCE_LEN, len(self.unique_words)))
            for t, word in enumerate(sentence):
                x_pred[0, t, self.word_indices[word]] = 1.

            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds)
            next_word = self.indices_word[next_index]

            generated += next_word
            sentence = sentence[1:] + next_word

            sys.stdout.write(next_word)
            sys.stdout.flush()
        print()


if __name__ == '__main__':
    runner = Trainer()
    runner.run()
