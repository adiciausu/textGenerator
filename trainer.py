from __future__ import print_function

import os

import keras
import tensorflow
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional, Activation, Embedding
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

import numpy as np
import io

class Trainer:
    EMBEDING_SIZE = 128

    model = None
    sequence_size = None
    vocabulary_size = None

    path = "big.txt"
    corpus_lines = []
    tokenizer = Tokenizer()

    def run(self):
        if not os.path.exists('data'):
            os.mkdir('data')
        if not os.path.exists(os.path.join('data', 'logs')):
            os.mkdir(os.path.join('data', 'logs'))
        if not os.path.exists(os.path.join('data', 'checkpoints')):
            os.mkdir(os.path.join('data', 'checkpoints'))

        config = tensorflow.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        keras.backend.set_session(tensorflow.Session(config=config))

        with io.open(self.path, encoding='utf-8') as f:
            self.corpus = f.read().lower()
            self.corpus_lines = self.corpus.split('\n')
            print('Corpus length in letters:', len(self.corpus))
            print('Corpus length in lines:', len(self.corpus_lines))

            self.tokenizer.fit_on_texts(self.corpus_lines)
            self.vocabulary_size = len(self.tokenizer.word_index) + 1
            print('Unique words:', self.vocabulary_size)

            sequences = []
            for line in self.corpus_lines:
                token_list = self.tokenizer.texts_to_sequences([line])[0]
                for i in range(10, len(token_list), 2):
                    n_gram_sequence = token_list[:i + 1]
                    sequences.append(n_gram_sequence)

            self.sequence_size = max([len(x) for x in sequences])
            sequences = pad_sequences(sequences, self.sequence_size, padding='pre')

            sequences = np.array(sequences)
            x, y = sequences[:, :-1], sequences[:, -1]
            y = to_categorical(y, num_classes=self.vocabulary_size)

            self.model = self.build_model()
            self.model.fit(x, y, validation_split=0.1, batch_size=128, epochs=60, callbacks=self.build_callbacks())

    def build_callbacks(self):
        checkpoint_path = "./data/checkpoints/text-generator-epoch{epoch:03d}-sequence%d-" \
                          "loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}" % self.sequence_size
        checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_acc', save_best_only=True)
        print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)
        early_stopping = EarlyStopping(monitor='val_acc', patience=100)
        tb_callback = TensorBoard(os.path.join('data', 'logs'))

        return [print_callback, checkpoint_callback, early_stopping, tb_callback]

    def build_model(self):
        model = Sequential()
        model = self.add_model_layers(model)
        print(model.summary())
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

        return model

    def add_model_layers(self, model):
        # Embedding
        model.add(Embedding(input_dim=self.vocabulary_size, output_dim=self.EMBEDING_SIZE, input_length=self.sequence_size - 1))
        # Hidden
        model.add(Bidirectional(LSTM(128)))
        model.add(Dropout(0.1))
        # Output
        model.add(Dense(self.vocabulary_size, activation='softmax'))

        return model

    def on_epoch_end(self, epoch, _):
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        seed = "One ball after another passed over as he approached and"

        for _ in range(20):
            token_list = self.tokenizer.texts_to_sequences([seed])[0]
            token_list = pad_sequences([token_list], maxlen=self.sequence_size - 1, padding='pre')
            predicted = self.model.predict_classes(token_list, verbose=0)

            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed += " " + output_word

        print(seed)

if __name__ == '__main__':
    runner = Trainer()
    runner.run()
