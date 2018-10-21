#!/usr/bin/env python3

import numpy as np
import time
import os
import sys
import shutil

from keras.datasets import cifar10

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.utils import to_categorical

def main():
    if len(sys.argv) < 2:
        print("Usage: './main.py <modelname> <number_of_samples_to_test>")
        exit()
    modelName = sys.argv[1]
    nSamples = int(sys.argv[2])
    modelFilename = '{}.h5'.format(modelName)

    print('Loading pre-trained model {}'.format(modelName))
    model = load_model(modelFilename)
    print(model.summary())

    if os.path.isdir(modelName):
        shutil.rmtree(modelName)

    os.makedirs(os.path.join(modelName, 'hit'), exist_ok=True)
    os.makedirs(os.path.join(modelName, 'miss'), exist_ok=True)

    _, (x_test, y_test) = cifar10.load_data()
    choices = np.random.choice(x_test.shape[0], nSamples, replace=False)
    x_test = x_test[choices]
    y_test = np.argmax(to_categorical(y_test[choices]), axis=1)

    proba = model.predict(x_test)
    guess = np.argmax(proba, axis=1)

    print(proba.shape, guess.shape, y_test.shape)

    for idx, (g, p, target) in enumerate(zip(guess, proba, y_test)):
        if g == target:
            print(' Hit! {}'.format(idx))
        else:
            print('Miss! {}'.format(idx))


if __name__ == '__main__':
    main()
