#!/usr/bin/env python3

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from Adaline import adaline

bands = [0, 1] # 0 for 0.01 and 1 for 0.05
muList = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
nDelaysList = [8, 32, 128]

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
    
        mu = 0.01
        fig = plt.figure(figsize=(8,4.5))
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        # plt.plot(t, u, label='input')
        plt.plot(t, y, label='signal')
        for nDelays in nDelaysList:
            yAdaptive, _, mseAdaptive, _ = adaline(u, y, mu=mu, nDelays=nDelays)
            if mseAdaptive < 100:
                plt.plot(t, yAdaptive, label='$L$ = {}'.format(nDelays))
        plt.xlim(-5, 125)
        plt.title('ADALINE behavior with fixed $\mu$={} varying $L$'.format(mu))
        plt.legend()
        plt.savefig('vary_nDelays_{}.png'.format(band), dpi=300)
        plt.close(fig)

        nDelays = 128
        fig = plt.figure(figsize=(8,4.5))
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        # plt.plot(t, u, label='input')
        plt.plot(t, y, label='signal')
        for mu in muList:
            yAdaptive, _, mseAdaptive, _ = adaline(u, y, mu=mu, nDelays=nDelays)
            if mseAdaptive < 100:
                plt.plot(t, yAdaptive, label='$\mu$ = {}'.format(mu))
        plt.xlim(-5, 125)
        plt.title('ADALINE behavior with fixed $L$={} varying $\mu$'.format(nDelays))
        plt.legend()
        plt.savefig('vary_mu_{}.png'.format(band), dpi=300)
        plt.close(fig)

        


if __name__ == '__main__':
    main()