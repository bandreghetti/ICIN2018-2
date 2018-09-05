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
    nEpochs=200
    nRemind=10
    nNoisy=15
    nNoisyTest=50
    
    origPattern = chars.getSymbols()
    origTarget = chars.getLabels()

    tenPattern = chars.getSymbols(noise=0.1)
    tenTarget = chars.getLabels()
    for _ in range(nNoisy-1):
        tenPattern = np.append(tenPattern, chars.getSymbols(noise=0.1), axis=0)
        tenTarget = np.append(tenTarget, chars.getLabels(), axis=0)    
    twentyPattern = chars.getSymbols(noise=0.2)
    twentyTarget = chars.getLabels()
    for _ in range(nNoisy-1):
        twentyPattern = np.append(twentyPattern, chars.getSymbols(noise=0.2), axis=0)
        twentyTarget = np.append(twentyTarget, chars.getLabels(), axis=0)
    thirtyPattern = chars.getSymbols(noise=0.3)
    thirtyTarget = chars.getLabels()
    for _ in range(nNoisy-1):
        thirtyPattern = np.append(thirtyPattern, chars.getSymbols(noise=0.3), axis=0)
        thirtyTarget = np.append(thirtyTarget, chars.getLabels(), axis=0)
    fortyPattern = chars.getSymbols(noise=0.4)
    fortyTarget = chars.getLabels()
    for _ in range(nNoisy-1):
        fortyPattern = np.append(fortyPattern, chars.getSymbols(noise=0.4), axis=0)
        fortyTarget = np.append(fortyTarget, chars.getLabels(), axis=0)
    fiftyPattern = chars.getSymbols(noise=0.5)
    fiftyTarget = chars.getLabels()
    for _ in range(nNoisy-1):
        fiftyPattern = np.append(fiftyPattern, chars.getSymbols(noise=0.5), axis=0)
        fiftyTarget = np.append(fiftyTarget, chars.getLabels(), axis=0)

    # Initialize 6 Keras models
    models = [None]*6
    for idx in range(6):
        models[idx] = Sequential()
        models[idx].add(Dense(units=40, activation='sigmoid', kernel_initializer="uniform", input_dim=63))
        models[idx].add(Dense(units=16, activation='sigmoid', kernel_initializer="uniform"))
        models[idx].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model using only the 16 original characters
    models[0].fit(origPattern, origTarget,
                  batch_size=2, epochs=nEpochs, verbose=2)

    # Copy model weights to the next one
    models[1].set_weights(models[0].get_weights())

    # Train next model with noisy characters and then remind it of the original pattern
    models[1].fit(tenPattern, tenTarget,
                  batch_size=8, epochs=nEpochs, verbose=2)
    models[1].fit(origPattern, origTarget,
                  batch_size=1, epochs=nRemind, verbose=2)

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
                  batch_size=8, epochs=nEpochs, verbose=2)
    models[2].fit(origPattern, origTarget,
                  batch_size=1, epochs=nRemind, verbose=2)

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
                  batch_size=8, epochs=nEpochs, verbose=2)
    models[3].fit(origPattern, origTarget,
                  batch_size=1, epochs=nRemind, verbose=2)

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
                  batch_size=8, epochs=nEpochs, verbose=2)
    models[4].fit(origPattern, origTarget,
                  batch_size=1, epochs=nRemind, verbose=2)

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
                  batch_size=8, epochs=nEpochs, verbose=2)
    models[5].fit(origPattern, origTarget,
                  batch_size=1, epochs=nRemind, verbose=2)

    # Plot model prediction of a single set of noisy characters
    sample = chars.getSymbols(noise=0.5)
    prediction = np.argmax(models[5].predict(sample), axis=1)
    chars.plotPrediction(sample, prediction, fileName='prediction50%.png')
    prediction = np.argmax(models[0].predict(sample), axis=1)
    chars.plotPrediction(sample, prediction, fileName='prediction50%_nonoise.png')

    # Generate new sample sets for evaluation of performance
    tenPattern = chars.getSymbols(noise=0.1)
    tenTarget = chars.getLabels()
    for _ in range(nNoisyTest-1):
        tenPattern = np.append(tenPattern, chars.getSymbols(noise=0.1), axis=0)
        tenTarget = np.append(tenTarget, chars.getLabels(), axis=0)
    twentyPattern = chars.getSymbols(noise=0.2)
    twentyTarget = chars.getLabels()
    for _ in range(nNoisyTest-1):
        twentyPattern = np.append(twentyPattern, chars.getSymbols(noise=0.2), axis=0)
        twentyTarget = np.append(twentyTarget, chars.getLabels(), axis=0)
    thirtyPattern = chars.getSymbols(noise=0.3)
    thirtyTarget = chars.getLabels()
    for _ in range(nNoisyTest-1):
        thirtyPattern = np.append(thirtyPattern, chars.getSymbols(noise=0.3), axis=0)
        thirtyTarget = np.append(thirtyTarget, chars.getLabels(), axis=0)
    fortyPattern = chars.getSymbols(noise=0.4)
    fortyTarget = chars.getLabels()
    for _ in range(nNoisyTest-1):
        fortyPattern = np.append(fortyPattern, chars.getSymbols(noise=0.4), axis=0)
        fortyTarget = np.append(fortyTarget, chars.getLabels(), axis=0)
    fiftyPattern = chars.getSymbols(noise=0.5)
    fiftyTarget = chars.getLabels()
    for _ in range(nNoisyTest-1):
        fiftyPattern = np.append(fiftyPattern, chars.getSymbols(noise=0.5), axis=0)
        fiftyTarget = np.append(fiftyTarget, chars.getLabels(), axis=0)

    # Evaluate performance of the network trained without noise for each sample    
    print('Model[0] error (00% noise): ' + str())
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

    x = np.array([0, 10, 20, 30, 40, 50])
    y0 = 100*np.array([ evalPerformance(models[0], origPattern, origTarget),
                        evalPerformance(models[0], tenPattern, tenTarget),
                        evalPerformance(models[0], twentyPattern, twentyTarget),
                        evalPerformance(models[0], thirtyPattern, thirtyTarget),
                        evalPerformance(models[0], fortyPattern, fortyTarget),
                        evalPerformance(models[0], fiftyPattern, fiftyTarget)
                      ])
    y1 = 100*np.array([ evalPerformance(models[1], origPattern, origTarget),
                        evalPerformance(models[1], tenPattern, tenTarget),
                        evalPerformance(models[1], twentyPattern, twentyTarget),
                        evalPerformance(models[1], thirtyPattern, thirtyTarget),
                        evalPerformance(models[1], fortyPattern, fortyTarget),
                        evalPerformance(models[1], fiftyPattern, fiftyTarget)
                      ])
    y2 = 100*np.array([ evalPerformance(models[2], origPattern, origTarget),
                        evalPerformance(models[2], tenPattern, tenTarget),
                        evalPerformance(models[2], twentyPattern, twentyTarget),
                        evalPerformance(models[2], thirtyPattern, thirtyTarget),
                        evalPerformance(models[2], fortyPattern, fortyTarget),
                        evalPerformance(models[2], fiftyPattern, fiftyTarget)
                      ])
    y3 = 100*np.array([ evalPerformance(models[3], origPattern, origTarget),
                        evalPerformance(models[3], tenPattern, tenTarget),
                        evalPerformance(models[3], twentyPattern, twentyTarget),
                        evalPerformance(models[3], thirtyPattern, thirtyTarget),
                        evalPerformance(models[3], fortyPattern, fortyTarget),
                        evalPerformance(models[3], fiftyPattern, fiftyTarget)
                      ])
    y4 = 100*np.array([ evalPerformance(models[4], origPattern, origTarget),
                        evalPerformance(models[4], tenPattern, tenTarget),
                        evalPerformance(models[4], twentyPattern, twentyTarget),
                        evalPerformance(models[4], thirtyPattern, thirtyTarget),
                        evalPerformance(models[4], fortyPattern, fortyTarget),
                        evalPerformance(models[4], fiftyPattern, fiftyTarget)
                      ])
    y5 = 100*np.array([ evalPerformance(models[5], origPattern, origTarget),
                        evalPerformance(models[5], tenPattern, tenTarget),
                        evalPerformance(models[5], twentyPattern, twentyTarget),
                        evalPerformance(models[5], thirtyPattern, thirtyTarget),
                        evalPerformance(models[5], fortyPattern, fortyTarget),
                        evalPerformance(models[5], fiftyPattern, fiftyTarget)
                      ])

    fig = plt.figure()
    plt.plot( x, y0, marker='', color='red', linewidth=1, label='0% Noise Training')
    plt.plot( x, y1, marker='', color='green', linewidth=1, label='10% Noise Training')
    plt.plot( x, y2, marker='', color='blue', linewidth=1, label='20% Noise Training')
    plt.plot( x, y3, marker='', color='cyan', linewidth=1, label='30% Noise Training')
    plt.plot( x, y4, marker='', color='magenta', linewidth=1, label='40% Noise Training')
    plt.plot( x, y5, marker='', color='yellow', linewidth=1, label='50% Noise Training')
    plt.title('Neural Network Test Error x Noise')
    plt.axis([0, 60, 0, 100])
    plt.xlabel('Sample Noise (%)')
    plt.ylabel('Error (%)')
    plt.grid(which='major')
    plt.minorticks_on()
    
    plt.legend()

    fig.savefig('results.png')
    plt.close(fig)


main()