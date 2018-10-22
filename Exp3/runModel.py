#!/usr/bin/env python3

import numpy as np
import time
import os
import sys
import shutil

from keras.datasets import cifar10

from keras.models import load_model

from keras.utils import to_categorical, print_summary

import matplotlib.pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

def main():
    if len(sys.argv) < 3:
        print("Usage: './main.py <modelname> <number_of_samples_to_test>")
        exit()
    modelName = sys.argv[1]
    nSamples = int(sys.argv[2])
    modelFilename = '{}.h5'.format(modelName)

    print('Loading pre-trained model {}'.format(modelName))
    model = load_model(modelFilename)

    # if os.path.isdir(modelName):
    #     shutil.rmtree(modelName)

    os.makedirs(os.path.join(modelName, 'hit'), exist_ok=True)
    os.makedirs(os.path.join(modelName, 'miss'), exist_ok=True)

    _, (x_test, y_test) = cifar10.load_data()

    x_test = x_test.astype('float64') / 255.0
    y_test = to_categorical(y_test)

    # summary_file = open(os.path.join(modelName, 'summary.txt'), 'w')
    # # summary_file.write()
    # scores = model.evaluate(x_test, y_test, verbose=0)
    # summary_file.write('Accuracy: {}\n\n'.format((scores[1]*100)))
    # print_summary(model, print_fn=lambda x: summary_file.write(x + '\n'))
    # summary_file.close()

    choices = np.random.choice(x_test.shape[0], nSamples, replace=False)
    x_test = x_test[choices]
    y_test = np.argmax(y_test[choices], axis=1)

    proba = model.predict(x_test)
    guess = np.argmax(proba, axis=1)
    sortIdx = np.argsort(proba, axis=1)

    classNames = np.array(['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

    for idx, (img, g, p, sort, target) in enumerate(zip(x_test, guess, proba, sortIdx, y_test)):
        fig, ax = plt.subplots()

        index = np.arange(1, 7)
        prob = p[sort][0:6]
        rects1 = ax.bar(index, prob, 0.7, color='b', alpha=0.6)
        classLabels = classNames[sort][0:6]
        ax.set_xticklabels(classLabels)
        plt.ylabel('Predicted probability', size=10)
        plt.ylim(0, 1)

        imagebox = OffsetImage(img, zoom=3.)
        imagebox.image.axes = ax

        ab = AnnotationBbox(imagebox, (1, 1),
                        xybox=(296, 0),
                        xycoords='data',
                        boxcoords="offset points",
                        box_alignment=(1., 1.),
                        pad=0
                        )

        ax.add_artist(ab)

        fig.tight_layout()
        plt.savefig(os.path.join(modelName, 'barchart.png'), dpi=300)

        if g == target:
            print(' Hit! {}'.format(idx))
        else:
            print('Miss! {}'.format(idx))

        plt.close(fig)


if __name__ == '__main__':
    main()
