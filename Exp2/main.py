#!/usr/bin/env python3

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from Adaline import adaline

bands = [0, 1] # 0 for 0.01 and 1 for 0.05
muList = [0.003, 0.01, 0.03, 0.1, 0.3]
nDelaysList = [8, 16, 32, 64, 128, 256]

def main():
    for band in bands:
        u = np.genfromtxt('prbs.csv', dtype=float, delimiter=',', skip_header=1).transpose()[band][0:1200]
        t = np.linspace(0,119.9,1200)
        G1 = signal.lti([1],
                        (1, 0.2, 1)) 
        G2 = signal.lti([3],
                        (1, 2, 1))
        
        t1, y1, x1 = signal.lsim(G1, u[0:800], t[0:800])
        t2, y2, _ = signal.lsim(G2, u[800:1200], t[800:1200], X0=x1[-1])

        t = np.concatenate((t1, t2))
        y = np.concatenate((y1, y2))
        
        for nDelays in nDelaysList:
            ########################################################
            
            # NEWLIND
            # y1 = w0*1 + w1*u1[0] + w2*u1[-1] + w3*u1[-2] + ...
            # y2 = w0*1 + w1*u2[0] + w2*u2[-1] + w3*u2[-2] + ...
            # ...
            # yN = w0*1 + w1*u2[0] + w2*u2[-1] + w3*u2[-2] + ...
            #
            #                               Y_Nx1   - Measurements
            # Y = U*Psi'                    U_NxM   - Regressors    
            # U'Y = U'U*Psi'                Psi_1xM - Parameter Vector
            # Psi'= (U'U)-1 U' Y            (U'U)-1 U' - Pseudoinverse - MMQ  

            # NEWLIND - Design a ADALINE using all samples at once: the pseudoinverse
            uMat = np.zeros((len(u), nDelays))
            for j in range(nDelays):         # build U with copies of shifted u
                uMat[j:len(uMat), j] = u[0:len(uMat)-j]

            #wd = inv(U'U)*U'*y
            wDesign = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(uMat), uMat)), np.transpose(uMat)), y)
            # wDesign = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(uMat),uMat)),np.transpose(uMat)),y)

            # NEWLIND Fixed weights wd - Designed
            yDesign, _, mseDesign = adaline(u, y, wDesign)

            ########################################################
            for mu in muList:
                yAdaptative, _, mseAdaptative, wHistoryAdaptative = adaline(u, y, mu=mu, nDelays=nDelays)
                yFixed, _, mseFixed = adaline(u, y, wHistoryAdaptative[-1, :])

                plt.figure(figsize=(16, 9))
                plt.xlim(-5, 125)
                plt.plot(t, u, label='input')
                plt.plot(t, y, label='signal')
                plt.plot(t, yAdaptative, label='adaptative err={:.2f}'.format(mseAdaptative))
                plt.plot(t, yFixed, label='fixed err={:.2f}'.format(mseFixed))
                plt.plot(t, yDesign, label='design err={:.2f}'.format(mseDesign))

                plt.legend()
                fileName = 'results_band={0}_nDelays={1:03d}_mu={2:.2f}.png'.format(band, nDelays, mu)
                try:
                    plt.savefig(fileName, dpi=300)
                except:
                    print('{} diverged too much, failed to plot'.format(fileName))

if __name__ == '__main__':
    main()