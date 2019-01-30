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
    SEQUENCE_LEN = 40
    STEP = 3
    text = None
    model = None
    unique_chars = None
    char_indices = None
    indices_char = None

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

        print('corpus length:', len(self.text))

        self.unique_chars = sorted(list(set(self.text)))
        print('total unique chars:', len(self.unique_chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.unique_chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.unique_chars))

        sequences = []
        next_chars = []
        for i in range(0, len(self.text) - self.SEQUENCE_LEN, self.STEP):
            sequences.append(self.text[i: i + self.SEQUENCE_LEN])
            next_chars.append(self.text[i + self.SEQUENCE_LEN])
        print('sentences count:', len(sequences))

        print('Vectorization...')
        x = np.zeros((len(sequences), self.SEQUENCE_LEN, len(self.unique_chars)), dtype=np.bool)
        y = np.zeros((len(sequences), len(self.unique_chars)), dtype=np.bool)
        for i, sequence in enumerate(sequences):
            for t, char in enumerate(sequence):
                x[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[next_chars[i]]] = 1

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
        model.add(Bidirectional(LSTM(128), input_shape=(self.SEQUENCE_LEN, len(self.unique_chars))))
        model.add(Dropout(0.2))
        model.add(Dense(len(self.unique_chars), activation='softmax'))
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
            x_pred = np.zeros((1, self.SEQUENCE_LEN, len(self.unique_chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, self.char_indices[char]] = 1.

            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds)
            next_char = self.indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


if __name__ == '__main__':
    runner = Trainer()
    runner.run()
