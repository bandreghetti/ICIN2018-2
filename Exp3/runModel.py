#!/usr/bin/env python3

import numpy as np
import time
import os
import sys
import shutil

from keras.datasets import cifar10

from keras.models import load_model, model_from_json

from keras.utils import to_categorical, print_summary

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def main():
    if len(sys.argv) < 3:
        print("Usage: './main.py <modelname> <number_of_samples_to_test>")
        exit()
    modelName = sys.argv[1]
    nSamples = int(sys.argv[2])

    print('Loading pre-trained model {}'.format(modelName))
    # load json and create model
    json_path = os.path.join(modelName, 'model.json')
    with open(json_path, 'r') as json_file:
        json_file = open(json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close() 
    model = model_from_json(loaded_model_json)
    
    # load weights into new model
    weightsPath = os.path.join(modelName, 'model.h5')
    model.load_weights(weightsPath)
    print("Loaded model from disk")

    hitFolder = os.path.join(modelName, 'hit')
    if os.path.isdir(hitFolder):
        shutil.rmtree(hitFolder)
    missFolder = os.path.join(modelName, 'miss')
    if os.path.isdir(missFolder):
        shutil.rmtree(missFolder)

    os.makedirs(os.path.join(modelName, 'hit'), exist_ok=True)
    os.makedirs(os.path.join(modelName, 'miss'), exist_ok=True)

    _, (x_test, y_test) = cifar10.load_data()

    x_test = x_test.astype('float64') / 255.0
    y_test = to_categorical(y_test)

    choices = np.random.choice(x_test.shape[0], nSamples, replace=False)
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
