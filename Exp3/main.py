#!/usr/bin/env python3

import numpy as np
import time

from keras.datasets import cifar100

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

def main():
    np.random.seed(int(time.time()))

    (x_train, y_train_idx), (x_test, y_test_idx) = cifar100.load_data(label_mode='fine')
    y_train = to_categorical(y_train_idx)
    y_test = to_categorical(y_test_idx)
    
    num_classes = y_test[0].size

    print(x_train[0].shape)

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

    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    model.fit(x_train, y_train, validation_data=(x_test, y_test), 
          epochs=1, batch_size=50, shuffle=True, callbacks=[earlyStopping])

if __name__ == '__main__':
    main()
