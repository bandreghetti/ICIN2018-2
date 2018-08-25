#!/usr/bin/env python3

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import chars
from matplotlib import pyplot as plt

def main():
    chars.plotsymbols(chars.symbols)
    # model = Sequential()

    # model.add(Dense(units=32, activation='sigmoid', input_dim=63))
    # model.add(Dense(units=24, activation='sigmoid'))
    # model.add(Dense(units=16, activation='sigmoid'))

    # model.compile(optimizer='adam', loss='mean_squared_error')

    # model.fit(chars.symbols, chars.labels, batch_size=1, epochs=500, verbose=2)

    # print(np.argmax(model.predict(chars.symbols), axis=1))

main()