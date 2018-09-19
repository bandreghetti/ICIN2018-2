#!/usr/bin/env python3

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from Adaline import adaline

band = 1 # 0 for 0.01 and 1 for 0.05
mu = 0.01
nDelays = 4

def main():
    t = np.linspace(0,119.9,1200)
    u = np.genfromtxt('prbs.csv', dtype=float, delimiter=',', skip_header=1).transpose()[band][0:1200]
    G1 = signal.lti([1],
                    (1, 0.2, 1)) 
    G2 = signal.lti([3],
                    (1, 2, 1))
    
    t1, y1, x1 = signal.lsim(G1, u[0:800], t[0:800])
    t2, y2, _ = signal.lsim(G2, u[800:1200], t[800:1200], X0=x1[-1])

    t = np.concatenate((t1, t2))
    y = np.concatenate((y1, y2))

    yOnline, eHistoryOnline, mseOnline, wHistoryOnline = adaline(u, y, mu=mu, nDelays=nDelays)
    yFixed, eHistoryFixed, mseFixed = adaline(u, y, wHistoryOnline[-1, :])

    # # NEWLIND
    # # y1 = w0*1 + w1*u1[0] + w2*u1[-1] + w3*u1[-2] + ...
    # # y2 = w0*1 + w1*u2[0] + w2*u2[-1] + w3*u2[-2] + ...
    # # ...
    # # yN = w0*1 + w1*u2[0] + w2*u2[-1] + w3*u2[-2] + ...
    # #
    # #                               Y_Nx1   - Measurements
    # # Y = U*Psi'                    U_NxM   - Regressors    
    # # U'Y = U'U*Psi'                Psi_1xM - Parameter Vector
    # # Psi'= (U'U)-1 U' Y            (U'U)-1 U' - Pseudoinverse - MMQ  

    # # NEWLIND - Design a ADALINE using all samples at once: the pseudoinverse
    # U=np.zeros((len(u)+delays,delays))
    # for j in range(delays):         # build U with copies of shifted u
    #     U[j:len(u)+j,j]=u

    # Y=np.zeros((len(u)+delays))
    # Y[0:len(y)]=y

    # #wd = inv(U'U)*U'*y
    # wd = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(U),U)),np.transpose(U)),Y)


    # # NEWLIND Fixed weights wd - Designed
    # yd, e, msed, wr =adaline(u,*wd)

    # ########################################################

    plt.figure(figsize=(16, 9))
    plt.xlim(-5, 125)
    plt.plot(t, u, label='input')
    plt.plot(t, y, label='signal')
    plt.plot(t, yOnline, label='adaptative')
    plt.plot(t, yFixed, label='fixed')
    plt.legend()
    plt.savefig('results_band={}_mu={}_nDelays={}.png'.format(band, mu, nDelays), dpi=300)

if __name__ == '__main__':
    main()