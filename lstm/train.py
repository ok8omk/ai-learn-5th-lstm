# https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

from __future__ import print_function
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--opt',\
        choices = ['rmsprop', 'adam', 'sgd'],\
        default = 'rmsprop')
parser.add_argument('--test',
        action = 'store_true')
args = parser.parse_args()

from keras import optimizers
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.callbacks import TensorBoard
import numpy as np
import random
import sys
import os
import json

learning_rate   = 0.001
iteration_num   = 2
epoch_num       = 20

sample_sentence = 'こんにちは'
maxlen = len(sample_sentence)

f_dict = 'dict'
c2i_path = os.path.join('.', f_dict, 'char_indices.json')
i2c_path = os.path.join('.', f_dict, 'indices_char.json')

f_model = 'model'
model_path = os.path.join('.', f_model, 'lstm_model.json')
param_path = os.path.join('.', f_model, 'lstm_model_weights_' + args.opt + '.hdf5')

opts = {
        'rmsprop'   : optimizers.RMSprop(learning_rate),
        'adam'      : optimizers.Adam(learning_rate),
        'sgd'       : optimizers.SGD(learning_rate)
        }

def is_hiragana(text):
    a = [ch for ch in text if "あ" <= ch <= "ん" or ch in ["「", "」"] ]
    if len(text) == len(a):
        return True
    return False

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def train():
    path = "./souseki_all_hiragana2.txt"
    text = open(path).read().lower()
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    with open(c2i_path, 'w') as f:
        json.dump(char_indices, f)
    with open(i2c_path, 'w') as f:
        json.dump(indices_char, f)

    # cut the text in semi-redundant sequences of maxlen characters
    input_sentence = sample_sentence
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=opts[args.opt])
    print('Optimizer:', args.opt)
    print('Learning Rate:', learning_rate)
    
    log_filepath = "./logs/" + args.opt + "/"
    tb_cb = TensorBoard(log_dir=log_filepath, write_graph=True, write_images=True)

    # train the model, output generated text after each iteration
    print()
    print('-' * 50)
    model.fit(X, y,
            batch_size=512,
            callbacks=[tb_cb],
            epochs=epoch_num)

    start_index = random.randint(0, len(text) - maxlen - 1)

    diversity = 0.2
    print()
    print('----- diversity:', diversity)

    generated = ''
    generated += input_sentence
    print('----- Generating with seed: "' + input_sentence + '"')
    sys.stdout.write(generated)

    for i in range(400):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()

    print('save the architecture of a model')
    json_string = model.to_json()
    open(model_path, 'w').write(json_string)
    print('save weights')
    model.save_weights(param_path)

def test():
    path = "souseki_all_hiragana2.txt"
    text = open(path).read().lower()
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))

    char_indices = {}
    indices_char = {}
    with open(c2i_path, 'r') as f:
        char_indices = json.load(f)
    with open(i2c_path, 'r') as f:
        indices_char = json.load(f)

    json_string = open(model_path).read()
    model = model_from_json(json_string)
    model.load_weights(param_path)

    print(maxlen, '文字のひらがなを入力してください')
    input_sentence = input()
    while not is_hiragana(input_sentence) and len(input_sentence) != maxlen:
        print(maxlen, '文字のひらがなを入力してください')
        input_sentence = input()
        
    generated = "「"
    generated += input_sentence + "」"
    input_sentence = "「" + input_sentence + "」"
    print('----- Generating with seed: "' + input_sentence + '"')
    sys.stdout.write(generated)

    for i in range(400):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(input_sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, 0.2)
        next_char = indices_char[str(next_index)]

        generated += next_char
        input_sentence = input_sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()

def response(input_sentence, opt):
    if input_sentence == '':
        return ''
    if not is_hiragana(input_sentence) or len(input_sentence) != maxlen:
        return str(maxlen) + '文字のひらがなを入力してください'
    path = "souseki_all_hiragana2.txt"
    text = open(path).read().lower()

    chars = sorted(list(set(text)))

    char_indices = {}
    indices_char = {}
    with open(c2i_path, 'r') as f:
        char_indices = json.load(f)
    with open(i2c_path, 'r') as f:
        indices_char = json.load(f)

    json_string = open(model_path).read()
    model = model_from_json(json_string)
    param_path = os.path.join('.', f_model, 'lstm_model_weights_' + opt + '.hdf5')
    model.load_weights(param_path)

    generated = "「"
    generated += input_sentence + "」"

    for i in range(400):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(input_sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, 0.2)
        next_char = indices_char[str(next_index)]

        generated += next_char
        input_sentence = input_sentence[1:] + next_char
    return generated

if __name__ == '__main__':
    print('Mode:', 'Test' if args.test else 'Train')
    if args.test:
        test()
    else:
        train()
