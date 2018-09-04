#!/usr/bin/env python3

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import keras.utils as kutils
import chars
from matplotlib import pyplot as plt

def evalPerformance(model, pattern, target):
    prediction = np.argmax(model.predict(pattern), axis=1)
    answer = np.argmax(target, axis=1)
    error = np.sum(np.not_equal(prediction, answer))/np.size(answer)
    return error

def main():
    origPattern = chars.getSymbols()
    origTarget = chars.getLabels()

    tenPattern = chars.getSymbols(noise=0.1)
    tenTarget = chars.getLabels()
    for _ in range(19):
        tenPattern = np.append(tenPattern, chars.getSymbols(noise=0.1), axis=0)
        tenTarget = np.append(tenTarget, chars.getLabels(), axis=0)    
    twentyPattern = chars.getSymbols(noise=0.2)
    twentyTarget = chars.getLabels()
    for _ in range(19):
        twentyPattern = np.append(twentyPattern, chars.getSymbols(noise=0.2), axis=0)
        twentyTarget = np.append(twentyTarget, chars.getLabels(), axis=0)
    thirtyPattern = chars.getSymbols(noise=0.3)
    thirtyTarget = chars.getLabels()
    for _ in range(19):
        thirtyPattern = np.append(thirtyPattern, chars.getSymbols(noise=0.3), axis=0)
        thirtyTarget = np.append(thirtyTarget, chars.getLabels(), axis=0)
    fortyPattern = chars.getSymbols(noise=0.4)
    fortyTarget = chars.getLabels()
    for _ in range(19):
        fortyPattern = np.append(fortyPattern, chars.getSymbols(noise=0.4), axis=0)
        fortyTarget = np.append(fortyTarget, chars.getLabels(), axis=0)
    fiftyPattern = chars.getSymbols(noise=0.5)
    fiftyTarget = chars.getLabels()
    for _ in range(19):
        fiftyPattern = np.append(fiftyPattern, chars.getSymbols(noise=0.5), axis=0)
        fiftyTarget = np.append(fiftyTarget, chars.getLabels(), axis=0)

    # Initialize 6 Keras models
    models = [None]*6
    for idx in range(6):
        models[idx] = Sequential()
        models[idx].add(Dense(units=24, activation='sigmoid', input_dim=63))
        models[idx].add(Dense(units=16, activation='linear'))
        models[idx].compile(optimizer='adam', loss='mean_squared_error')

    # Train model using only the 16 original characters
    models[0].fit(origPattern, origTarget,
                  batch_size=2, epochs=1000, verbose=2)

    # Copy model weights to the next one
    models[1].set_weights(models[0].get_weights())

    # Train next model with noisy characters and then remind it of the original pattern
    models[1].fit(tenPattern, tenTarget,
                  batch_size=8, epochs=1000, verbose=2)
    models[1].fit(origPattern, origTarget,
                  batch_size=1, epochs=30, verbose=2)

    # Plot model prediction of a single set of noisy characters
    sample = chars.getSymbols(noise=0.1)
    prediction = np.argmax(models[1].predict(sample), axis=1)
    chars.plotPrediction(sample, prediction, fileName='prediction10%.png')
    prediction = np.argmax(models[0].predict(sample), axis=1)
    chars.plotPrediction(sample, prediction, fileName='prediction10%_nonoise.png')

    # Copy model weights to the next one
    models[2].set_weights(models[1].get_weights())

    # Train next model with noisy characters and then remind it of the original pattern
    models[2].fit(twentyPattern, twentyTarget,
                  batch_size=8, epochs=1000, verbose=2)
    models[2].fit(origPattern, origTarget,
                  batch_size=1, epochs=30, verbose=2)

    # Plot model prediction of a single set of noisy characters
    sample = chars.getSymbols(noise=0.2)
    prediction = np.argmax(models[2].predict(sample), axis=1)
    chars.plotPrediction(sample, prediction, fileName='prediction20%.png')
    prediction = np.argmax(models[0].predict(sample), axis=1)
    chars.plotPrediction(sample, prediction, fileName='prediction20%_nonoise.png')

    # Copy model weights to the next one
    models[3].set_weights(models[2].get_weights())

    # Train next model with noisy characters and then remind it of the original pattern
    models[3].fit(thirtyPattern, thirtyTarget,
                  batch_size=8, epochs=1000, verbose=2)
    models[3].fit(origPattern, origTarget,
                  batch_size=1, epochs=30, verbose=2)

    # Plot model prediction of a single set of noisy characters
    sample = chars.getSymbols(noise=0.3)
    prediction = np.argmax(models[3].predict(sample), axis=1)
    chars.plotPrediction(sample, prediction, fileName='prediction30%.png')
    prediction = np.argmax(models[0].predict(sample), axis=1)
    chars.plotPrediction(sample, prediction, fileName='prediction30%_nonoise.png')

    # Copy model weights to the next one
    models[4].set_weights(models[3].get_weights())

    # Train next model with noisy characters and then remind it of the original pattern
    models[4].fit(fortyPattern, fortyTarget,
                  batch_size=8, epochs=1000, verbose=2)
    models[4].fit(origPattern, origTarget,
                  batch_size=1, epochs=30, verbose=2)

    # Plot model prediction of a single set of noisy characters
    sample = chars.getSymbols(noise=0.4)
    prediction = np.argmax(models[4].predict(sample), axis=1)
    chars.plotPrediction(sample, prediction, fileName='prediction40%.png')
    prediction = np.argmax(models[0].predict(sample), axis=1)
    chars.plotPrediction(sample, prediction, fileName='prediction40%_nonoise.png')

    # Copy model weights to the next one
    models[5].set_weights(models[4].get_weights())

    # Train next model with noisy characters and then remind it of the original pattern
    models[5].fit(fiftyPattern, fiftyTarget,
                  batch_size=8, epochs=1000, verbose=2)
    models[5].fit(origPattern, origTarget,
                  batch_size=1, epochs=30, verbose=2)

    # Plot model prediction of a single set of noisy characters
    sample = chars.getSymbols(noise=0.5)
    prediction = np.argmax(models[5].predict(sample), axis=1)
    chars.plotPrediction(sample, prediction, fileName='prediction50%.png')
    prediction = np.argmax(models[0].predict(sample), axis=1)
    chars.plotPrediction(sample, prediction, fileName='prediction50%_nonoise.png')

    # Generate new sample sets for evaluation of performance
    tenPattern = chars.getSymbols(noise=0.1)
    tenTarget = chars.getLabels()
    for _ in range(39):
        tenPattern = np.append(tenPattern, chars.getSymbols(noise=0.1), axis=0)
        tenTarget = np.append(tenTarget, chars.getLabels(), axis=0)
    twentyPattern = chars.getSymbols(noise=0.2)
    twentyTarget = chars.getLabels()
    for _ in range(39):
        twentyPattern = np.append(twentyPattern, chars.getSymbols(noise=0.2), axis=0)
        twentyTarget = np.append(twentyTarget, chars.getLabels(), axis=0)
    thirtyPattern = chars.getSymbols(noise=0.3)
    thirtyTarget = chars.getLabels()
    for _ in range(39):
        thirtyPattern = np.append(thirtyPattern, chars.getSymbols(noise=0.3), axis=0)
        thirtyTarget = np.append(thirtyTarget, chars.getLabels(), axis=0)
    fortyPattern = chars.getSymbols(noise=0.4)
    fortyTarget = chars.getLabels()
    for _ in range(39):
        fortyPattern = np.append(fortyPattern, chars.getSymbols(noise=0.4), axis=0)
        fortyTarget = np.append(fortyTarget, chars.getLabels(), axis=0)
    fiftyPattern = chars.getSymbols(noise=0.5)
    fiftyTarget = chars.getLabels()
    for _ in range(39):
        fiftyPattern = np.append(fiftyPattern, chars.getSymbols(noise=0.5), axis=0)
        fiftyTarget = np.append(fiftyTarget, chars.getLabels(), axis=0)

    # Evaluate performance of the network trained without noise for each sample
    print('Model[0] error (00% noise): ' + str(evalPerformance(models[0], origPattern, origTarget)))
    print('Model[0] error (10% noise): ' + str(evalPerformance(models[0], tenPattern, tenTarget)))
    print('Model[0] error (20% noise): ' + str(evalPerformance(models[0], twentyPattern, twentyTarget)))
    print('Model[0] error (30% noise): ' + str(evalPerformance(models[0], thirtyPattern, thirtyTarget)))
    print('Model[0] error (40% noise): ' + str(evalPerformance(models[0], fortyPattern, fortyTarget)))
    print('Model[0] error (50% noise): ' + str(evalPerformance(models[0], fiftyPattern, tenTarget)))

    print('Model[1] error (00% noise): ' + str(evalPerformance(models[1], origPattern, origTarget)))
    print('Model[1] error (10% noise): ' + str(evalPerformance(models[1], tenPattern, tenTarget)))
    print('Model[1] error (20% noise): ' + str(evalPerformance(models[1], twentyPattern, twentyTarget)))
    print('Model[1] error (30% noise): ' + str(evalPerformance(models[1], thirtyPattern, thirtyTarget)))
    print('Model[1] error (40% noise): ' + str(evalPerformance(models[1], fortyPattern, fortyTarget)))
    print('Model[1] error (50% noise): ' + str(evalPerformance(models[1], fiftyPattern, tenTarget)))

    print('Model[2] error (00% noise): ' + str(evalPerformance(models[2], origPattern, origTarget)))
    print('Model[2] error (10% noise): ' + str(evalPerformance(models[2], tenPattern, tenTarget)))
    print('Model[2] error (20% noise): ' + str(evalPerformance(models[2], twentyPattern, twentyTarget)))
    print('Model[2] error (30% noise): ' + str(evalPerformance(models[2], thirtyPattern, thirtyTarget)))
    print('Model[2] error (40% noise): ' + str(evalPerformance(models[2], fortyPattern, fortyTarget)))
    print('Model[2] error (50% noise): ' + str(evalPerformance(models[2], fiftyPattern, tenTarget)))

    print('Model[3] error (00% noise): ' + str(evalPerformance(models[3], origPattern, origTarget)))
    print('Model[3] error (10% noise): ' + str(evalPerformance(models[3], tenPattern, tenTarget)))
    print('Model[3] error (20% noise): ' + str(evalPerformance(models[3], twentyPattern, twentyTarget)))
    print('Model[3] error (30% noise): ' + str(evalPerformance(models[3], thirtyPattern, thirtyTarget)))
    print('Model[3] error (40% noise): ' + str(evalPerformance(models[3], fortyPattern, fortyTarget)))
    print('Model[3] error (50% noise): ' + str(evalPerformance(models[3], fiftyPattern, tenTarget)))

    print('Model[4] error (00% noise): ' + str(evalPerformance(models[4], origPattern, origTarget)))
    print('Model[4] error (10% noise): ' + str(evalPerformance(models[4], tenPattern, tenTarget)))
    print('Model[4] error (20% noise): ' + str(evalPerformance(models[4], twentyPattern, twentyTarget)))
    print('Model[4] error (30% noise): ' + str(evalPerformance(models[4], thirtyPattern, thirtyTarget)))
    print('Model[4] error (40% noise): ' + str(evalPerformance(models[4], fortyPattern, fortyTarget)))
    print('Model[4] error (50% noise): ' + str(evalPerformance(models[4], fiftyPattern, tenTarget)))

    print('Model[5] error (00% noise): ' + str(evalPerformance(models[5], origPattern, origTarget)))
    print('Model[5] error (10% noise): ' + str(evalPerformance(models[5], tenPattern, tenTarget)))
    print('Model[5] error (20% noise): ' + str(evalPerformance(models[5], twentyPattern, twentyTarget)))
    print('Model[5] error (30% noise): ' + str(evalPerformance(models[5], thirtyPattern, thirtyTarget)))
    print('Model[5] error (40% noise): ' + str(evalPerformance(models[5], fortyPattern, fortyTarget)))
    print('Model[5] error (50% noise): ' + str(evalPerformance(models[5], fiftyPattern, tenTarget)))


main()