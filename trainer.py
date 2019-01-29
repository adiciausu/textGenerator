from __future__ import print_function
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional
from keras.layers import LSTM
import numpy as np
import random
import sys
import io

class Trainer:
    SEQUENCE_LEN = 40
    STEP = 3
    text = None
    model = None
    chars = None
    char_indices = None
    indices_char = None

    def run(self):
        with io.open('macanache.txt', encoding='utf-8') as f:
            self.text = f.read().lower()
        print('corpus length:', len(self.text))

        self.chars = sorted(list(set(self.text)))
        print('total chars:', len(self.chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

        # cut the text in semi-redundant sequences of maxlen characters
        sentences = []
        next_chars = []
        for i in range(0, len(self.text) - self.SEQUENCE_LEN, self.STEP):
            sentences.append(self.text[i: i + self.SEQUENCE_LEN])
            next_chars.append(self.text[i + self.SEQUENCE_LEN])
        print('nb sequences:', len(sentences))

        print('Vectorization...')
        x = np.zeros((len(sentences), self.SEQUENCE_LEN, len(self.chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[next_chars[i]]] = 1

        self.model = self.build_model()
        self.model.fit(x, y, validation_split=0.33, batch_size=128, epochs=60, callbacks=self.build_callbacks())

    def build_callbacks(self):
        checkpoint_path = "./checkpoints/LSTM_LYRICS-epoch{epoch:03d}-sequence%d-" \
                          "loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}" % self.SEQUENCE_LEN
        checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_acc', save_best_only=True)
        print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)
        early_stopping = EarlyStopping(monitor='val_acc', patience=5)
        tb_callback = TensorBoard('logs')

        return [print_callback, checkpoint_callback, early_stopping, tb_callback]

    def build_model(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(128), input_shape=(self.SEQUENCE_LEN, len(self.chars))))
        model.add(Dropout(0.2))
        model.add(Dense(len(self.chars), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)

        return np.argmax(probas)

    def on_epoch_end(self, epoch, _):
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(self.text) - self.SEQUENCE_LEN - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = self.text[start_index: start_index + self.SEQUENCE_LEN]
            generated += sentence
            print('----- Generating with seed: "' + sentence.strip() + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, self.SEQUENCE_LEN, len(self.chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, self.char_indices[char]] = 1.

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_char = self.indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()


if __name__ == '__main__':
    runner = Trainer()
    runner.run()
