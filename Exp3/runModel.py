#!/usr/bin/env python3

import numpy as np
import time
import os
import sys
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
    modelFilename = '{}.h5'.format(modelName)

    print('Loading pre-trained model {}'.format(modelName))
    model = load_model(modelFilename)

    if os.path.isdir(modelName):
        os.rmdir(modelName)
    os.makedirs(os.path.join(modelName, exist_ok=True))
    print(model.summary())

    _, (x_test, y_test) = cifar10.load_data()
    choices = np.random.choice(x_test.shape[0], 3, replace=False)
    x_test = x_test[choices]
    y_test = to_categorical(y_test[choices])

    for idx, img, target in enumerate(zip(x_test, y_test)):
        model.predict_proba(img)
        guess = model.predict(img)
        if guess == target:
            print('Hit! {}'.format(idx))
        else:
            print('Miss! {}'.format(idx))


if __name__ == '__main__':
    main()
