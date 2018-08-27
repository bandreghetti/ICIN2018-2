#!/usr/bin/env python3

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import chars
from matplotlib import pyplot as plt

def main():
    pattern = chars.getSymbols()
    target = chars.getLabels()
    for _ in range(20):
        pattern = np.append(pattern, chars.getSymbols(noise=0.1), axis=0)
        target = np.append(target, chars.getLabels(), axis=0)
    for _ in range(20):
        pattern = np.append(pattern, chars.getSymbols(noise=0.2), axis=0)
        target = np.append(target, chars.getLabels(), axis=0)
    for _ in range(20):
        pattern = np.append(pattern, chars.getSymbols(noise=0.3), axis=0)
        target = np.append(target, chars.getLabels(), axis=0)

    model = Sequential()

    model.add(Dense(units=32, activation='sigmoid', input_dim=63))
    model.add(Dense(units=24, activation='sigmoid'))
    model.add(Dense(units=16, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(pattern, target, batch_size=16, epochs=1000, verbose=2)

    sample = chars.getSymbols(0.1)
    prediction = np.argmax(model.predict(sample), axis=1)
    chars.plotPrediction(sample, prediction)

    sample = chars.getSymbols(0.2)
    prediction = np.argmax(model.predict(sample), axis=1)
    chars.plotPrediction(sample, prediction)

    sample = chars.getSymbols(0.3)
    prediction = np.argmax(model.predict(sample), axis=1)
    chars.plotPrediction(sample, prediction)


main()