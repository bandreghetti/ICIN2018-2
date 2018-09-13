from keras.models import Sequential
from keras.layers import Dense
import numpy as np

class MultilayerPerceptron():
    def __init__(self, obsSpace):
        nPixels = obsSpace.sample().size
        self.model = Sequential()
        self.model.add(Dense(units=400, activation='sigmoid', kernel_initializer="uniform", input_dim=nPixels))
        self.model.add(Dense(units=20, activation='sigmoid', kernel_initializer="uniform"))
        self.model.add(Dense(units=5, activation='sigmoid', kernel_initializer="uniform"))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, observation, action):
        flatScr = observation.flatten()
        print(flatScr.shape)
        self.model.train_on_batch(flatScr, action)

    def action(self, observation):
        flatScr = observation.flatten()
        print(flatScr.shape)
        output = self.model.predict(flatScr)
        return output

