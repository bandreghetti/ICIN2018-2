from keras.models import Sequential, clone_model
from keras.layers import Dense
import numpy as np
from threading import Thread

class MultilayerPerceptron():
    def __init__(self, obsSpace):
        self.nPixels = obsSpace.sample().size
        self.trainModel = Sequential()
        self.trainModel.add(Dense(units=40, activation='sigmoid', kernel_initializer="uniform", input_dim=self.nPixels))
        self.trainModel.add(Dense(units=20, activation='sigmoid', kernel_initializer="uniform"))
        self.trainModel.add(Dense(units=5, activation='sigmoid', kernel_initializer="uniform"))
        self.trainModel.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        self.agentModel = clone_model(self.trainModel)
        self.currBatchSize = 0
        self.observationBatch = np.empty((0, self.nPixels))
        self.actionBatch = np.empty((0, 5))
        self.trainThread = None

    def train(self, observation, action):
        flatScr = np.expand_dims(observation.flatten(), 0)
        action = np.expand_dims(action, 0)
        self.observationBatch = np.append(self.observationBatch, flatScr, 0)
        self.actionBatch = np.append(self.actionBatch, action, 0)
        self.currBatchSize += 1
        if self.currBatchSize > 5:
            observationBatch = np.copy(self.observationBatch)
            actionBatch = np.copy(self.actionBatch)
            self.currBatchSize = 0
            self.observationBatch = np.empty((0, self.nPixels))    
            self.actionBatch = np.empty((0, 5))
            self.trainModel.train_on_batch(observationBatch, actionBatch)

    def action(self, observation):
        flatScr = np.expand_dims(observation.flatten(), 0)
        output = self.agentModel.predict(flatScr)
        return np.greater(output, 0.5).astype(np.float64)[0]
