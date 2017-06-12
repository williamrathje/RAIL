from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Merge
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import codecs

path = 'training.txt'
train = codecs.open(path, 'r', "utf-8").read().lower()
chars = sorted(list(set(train)))
print('total words:', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
maxlen = 40


train = train[0:int(.7*len(train))]
test = train[int(.7*len(train))+1:len(train)]

print('corpus length:', len(train))


def prepare(text):
    # cut the text in semi-redundant sequences of maxlen characters
    step = 3
    sentences = []
    sentences_post = []
    next_chars = []
    for i in range(0, len(text) - maxlen - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
        sentences_post.append(text[i+maxlen+1:i+maxlen+1+maxlen])
    print('nb sequences:', len(sentences))


    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    X_rev = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
        for t, char in enumerate(sentences_post[i]):
            X_rev[i, t, char_indices[char]] = 1
    return X, X_rev, y


# build the model: a single LSTM
print('Build model...')
left = Sequential()
left.add(LSTM(128, return_sequences=True, input_shape=(maxlen, len(chars))))
right = Sequential()
right.add(LSTM(128, go_backwards=True, return_sequences=True, input_shape=(maxlen, len(chars))))
model = Sequential()
model.add(Merge([left, right], mode='concat'))
model.add(Dropout(0.2))

left = Sequential()
right = Sequential()
left.add(model)
right.add(model)

left.add(LSTM(128, return_sequences=False, input_shape=(maxlen, len(chars))))
right.add(LSTM(128, go_backwards=True, return_sequences=False, input_shape=(maxlen, len(chars))))
model = Sequential()
model.add(Merge([left, right], mode='concat'))
model.add(Dropout(0.2))

model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 100):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    stp = 5000000
    #VAL_SPLIT_SIZE = 0.3
    i=0
    hist = None
    while (i+1)*stp < len(train):
        X, X_rev, y = prepare(train[i*stp: (i+1)*stp])
        model.fit([X, X_rev], y, batch_size=64, nb_epoch=1, verbose=2)
        i += 1
    if (i+1)*stp > len(train) and i*stp < len(train):
        X, X_rev, y = prepare(train[i*stp: len(train)])
        hist = model.fit([X, X_rev], y, batch_size=64, nb_epoch=1, verbose=2)

    print()
    print("Finished epoch: " + str(iteration))
    print(hist.history)

    print()
    test_X, test_X_rev, test_y = prepare(test)
    e = model.evaluate([test_X, test_X_rev], test_y, verbose=1)
    print(str(model.metrics_names))
    for x in e:
        print("Metric: " + str(x))

    print()

    #generate

    start_index = random.randint(0, len(train) - maxlen - 1)

    #for diversity in [0.2, 0.5, 1.0, 1.2]:
    for diversity in [0.5]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = train[start_index: start_index + maxlen]
        sentence_rev = train[start_index + maxlen+1:start_index + maxlen+1+maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            x_rev = np.zeros((1, maxlen, len(chars)))

            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.
            for t, char in enumerate(sentence_rev):
                x_rev[0, t, char_indices[char]] = 1.

            preds = model.predict([x,x_rev], verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
    #if iteration == 30 and (i+1)*stp >= (len(train)/stp): 
    model.save('model.h5')