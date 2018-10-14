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
        exit()
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
        # This is the basic model very similar to the one available in the example
        if modelName == 'model1':
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
        # This is a network with bigger kernels and more filters. Will probably take more time to train
        elif modelName == 'model2':
            model.add(Conv2D(128, (5, 5), input_shape=x_train[0].shape, padding='same', activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            model.add(Conv2D(512, (5, 5), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            model.add(Flatten())
            model.add(Dense(4096, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(2048, activation='relu'))
            model.add(Dropout(0.5))
        # This network uses no Dropout and only valid pixels. Maybe it works better.
        elif modelName == 'model3':
            model.add(Conv2D(64, (3, 3), input_shape=x_train[0].shape, padding='valid', activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            model.add(Conv2D(256, (3, 3), activation='relu', padding='valid'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

            model.add(Flatten())
            model.add(Dense(2048, activation='relu'))
            model.add(Dense(1024, activation='relu'))
        else:
            print('Unknown model {}. Exiting.'.format(modelName))
            exit()

        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        model.save(modelFilename)

    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    print('Training {} for {} epochs'.format(modelName, nEpochs))

    model.fit(x_train, y_train, validation_data=(x_test, y_test), 
          epochs=nEpochs, batch_size=50, shuffle=True, callbacks=[earlyStopping])

    model.save(modelFilename)

if __name__ == '__main__':
    main()
