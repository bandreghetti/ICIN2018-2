import numpy as np

def adaline(u, y, w=[], nDelays=4, mu=0.1):
    if len(w) == 0:
        adapt = True   # adapt weights w
    else:
        adapt = False   # receive and mantain w - simulate only

    uPad = np.insert(u, 0, np.zeros(nDelays-1))

    if adapt:
        w = np.zeros(nDelays)
        wHistory = np.zeros((len(u), nDelays))
    yOut = np.zeros(len(u))
    eHistory = np.zeros(len(u))
    errSum = np.array(0, dtype=np.float64)

    for i in range(len(u)):
        k = i+nDelays-1
        for j in range(nDelays):
            yOut[i] += uPad[k-j] * w[j]
        err = y[i] - yOut[i]
        eHistory[i] = err
        errSum += err**2
        if adapt==1:
            for j in range(nDelays): 
                w[j] += mu * err * uPad[k-j]               # Widrow-Hoff delta rule
                wHistory[i,j] = w[j]                       # store w history
    
    mse = errSum/len(u)
    
    if adapt:
        return yOut, eHistory, mse, wHistory
    else:
        return yOut, eHistory, mse