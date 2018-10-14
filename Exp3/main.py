#!/usr/bin/env python3

import numpy as np
import time
import os
import sys

from keras.datasets import cifar100

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

def main():
    if len(sys.argv) < 3:
        print("Usage: './main.py <modelname> <n_epochs>")

    modelName = sys.argv[1]
    nEpochs = int(sys.argv[2])

    modelFilename = '{}.h5'.format(modelName)
    
    np.random.seed(int(time.time()))

    (x_train, y_train_idx), (x_test, y_test_idx) = cifar100.load_data(label_mode='fine')
    y_train = to_categorical(y_train_idx)
    y_test = to_categorical(y_test_idx)
    
    num_classes = y_test[0].size

    if os.path.isfile(modelFilename):
        print('Loading pre-trained model {}'.format(modelName))
        model = load_model(modelFilename)
    else:
        print('Creating model {}'.format(modelName))
        model = Sequential()

        model.add(Conv2D(64, (3, 3), input_shape=x_train[0].shape, padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.save(modelFilename)

    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    model.fit(x_train, y_train, validation_data=(x_test, y_test), 
          epochs=nEpochs, batch_size=50, shuffle=True, callbacks=[earlyStopping])

    model.save(modelFilename)

if __name__ == '__main__':
    main()
