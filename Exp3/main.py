#!/usr/bin/env python3

import numpy as np
import time
import os
import sys

from keras.datasets import cifar10

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

    (x_train, y_train_idx), (x_test, y_test_idx) = cifar10.load_data()

    x_train = x_train.astype('float64') / 255.0
    x_test = x_test.astype('float64') / 255.0

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
        if modelName == 'model0':
            model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dropout(0.2))
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.2))
        # This is a network with higher dropout rate
        elif modelName == 'model1':
            model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
            model.add(Dropout(0.5))
            model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(Dropout(0.5))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(Dropout(0.5))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dropout(0.5))
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.5))
        # This network uses no Dropout
        elif modelName == 'model2':
            model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
            model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(1024, activation='relu'))
            model.add(Dense(512, activation='relu'))
        # This network uses striding when maxpooling the filters
        elif modelName == 'model3':
            model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Flatten())
            model.add(Dropout(0.2))
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.2))
        # This network convolutes only valid pixels
        elif modelName == 'model4':
            model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='valid'))
            model.add(Dropout(0.2))
            model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
            model.add(Dropout(0.2))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
            model.add(Dropout(0.2))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dropout(0.2))
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.2))
        # This network has no convolutional layers
        elif modelName == 'model5':
            model.add(Flatten())
            model.add(Dense(1024, input_shape=(3072,), activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.2))
        # This model uses 5x5 kernels instead of 3x3
        elif modelName == 'model6':
            model.add(Conv2D(32, (5, 5), input_shape=(32, 32, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dropout(0.2))
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.2))
        elif modelName == 'model7':
            model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dropout(0.2))
            model.add(Dense(2048, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.2))
        # This network has twice the number of filters
        elif modelName == 'model8':
            model.add(Conv2D(64, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dropout(0.2))
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.2))
        # This network uses no maxpooling
        elif modelName == 'model9':
            model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(Dropout(0.2))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
            model.add(Flatten())
            model.add(Dropout(0.2))
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.2))
        else:
            print('Unknown model {}. Exiting.'.format(modelName))
            exit()

        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(model.summary)

        model.save(modelFilename)

    earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

    print('Training {} for {} epochs'.format(modelName, nEpochs))

    model.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=nEpochs, batch_size=128, shuffle=True,
          callbacks=[earlyStopping], verbose=2)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    model.save(modelFilename)

if __name__ == '__main__':
    main()
