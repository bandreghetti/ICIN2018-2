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

from keras.callbacks import EarlyStopping
from keras.utils import to_categorical, print_summary

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def main():
    if len(sys.argv) < 3:
        print("Usage: './main.py <modelname> <n_epochs>")
        exit()
    modelName = sys.argv[1]
    nEpochs = int(sys.argv[2])

    if os.path.isdir(modelName):
        shutil.rmtree(modelName)

    modelFilename = '{}.h5'.format(modelName)

    np.random.seed(int(time.time()))

    (x_train, y_train_idx), (x_test, y_test_idx) = cifar10.load_data()

    x_train = x_train.astype('float64') / 255.0
    x_test = x_test.astype('float64') / 255.0

    y_train = to_categorical(y_train_idx)
    y_test = to_categorical(y_test_idx)

    num_classes = y_test[0].size

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
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
    # This network has no convolutional layers
    elif modelName == 'model5':
        model.add(Reshape((3072, ), input_shape=(32, 32, 3)))
        model.add(Dense(1024, activation='relu'))
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

    earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

    print('Training {} for {} epochs'.format(modelName, nEpochs))

    hist = model.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=nEpochs, batch_size=128, shuffle=True,
          callbacks=[earlyStopping], verbose=2)

    # From here on, save all data about the training and network performance

    os.makedirs(os.path.join(modelName, 'hit'), exist_ok=True)
    os.makedirs(os.path.join(modelName, 'miss'), exist_ok=True)

    np.save(os.path.join(modelName, 'train_history'), hist.history)

    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(modelName, 'model.json'), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(modelName, 'model.h5'))

    scores = model.evaluate(x_test, y_test, verbose=0)
    with open(os.path.join(modelName, 'summary.txt'), 'w') as summary_file:
        summary_file.write('Accuracy: {}\n\n'.format((scores[1]*100)))
        print_summary(model, print_fn=lambda x: summary_file.write(x + '\n'))
        summary_file.close()

    fig = plt.figure(figsize=(15,8))
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('{} accuracy'.format(modelName), size=20)
    plt.ylabel('Accuracy', size=20)
    plt.xlabel('Epoch', size=20)
    plt.legend(['train', 'test'], loc='upper left', prop={'size':15})
    plt.savefig(os.path.join(modelName, 'accuracy_history.png'), dpi=300)
    plt.close(fig)

    print("Saved model to disk")

    # Generate barcharts
    choices = np.random.choice(x_test.shape[0], 200, replace=False)
    x_test = x_test[choices]
    y_test = np.argmax(y_test[choices], axis=1)

    proba = model.predict(x_test)
    guess = np.argmax(proba, axis=1)
    sortIdx = np.flip(np.argsort(proba, axis=1), axis=1)

    classNames = np.array(['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

    for idx, (img, g, p, sort, target) in enumerate(zip(x_test, guess, proba, sortIdx, y_test)):
        fig, ax = plt.subplots()

        index = np.arange(1, 6)
        prob = p[sort][0:5]
        _ = ax.bar(index, prob, 0.7, color='b', alpha=0.6)
        classLabels = np.append('', classNames[sort])
        ax.set_xticklabels(classLabels)
        plt.title(classNames[target])
        plt.ylabel('Predicted probability', size=10)
        plt.ylim(0, 1)

        imagebox = OffsetImage(img, zoom=4.)
        imagebox.image.axes = ax

        ab = AnnotationBbox(imagebox, (1, 1),
                        xybox=(354, 0),
                        xycoords='data',
                        boxcoords="offset points",
                        box_alignment=(1., 1.),
                        pad=0
                        )

        ax.add_artist(ab)

        fig.tight_layout()
        if g == target:
            folder = os.path.join(modelName, 'hit')
        else:
            folder = os.path.join(modelName, 'miss')
        plt.savefig(os.path.join(folder, '{}.png'.format(idx)), dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    main()
