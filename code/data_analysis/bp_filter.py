# -*- coding: utf-8 -*-

import scipy

def bandpass_ifft(X, Low_cutoff, High_cutoff, F_sample, M=None):
    """Bandpass filtering on a real signal using inverse FFT
    
    Inputs
    =======
    
    X: 1-D numpy array of floats, the real time domain signal (time series) to be filtered
    Low_cutoff: float, frequency components below this frequency will not pass the filter (physical frequency in unit of Hz)
    High_cutoff: float, frequency components above this frequency will not pass the filter (physical frequency in unit of Hz)
    F_sample: float, the sampling frequency of the signal (physical frequency in unit of Hz)    
    
    Notes
    =====
    1. The input signal must be real, not imaginary nor complex
    2. The Filtered_signal will have only half of original amplitude. Use abs() to restore. 
    3. In Numpy/Scipy, the frequencies goes from 0 to F_sample/2 and then from negative F_sample to 0. 
    
    """        
    

    if M == None: # if the number of points for FFT is not specified
        M = X.size # let M be the length of the time series
    Spectrum = scipy.fft(X, n=M) 
    [Low_cutoff, High_cutoff, F_sample] = map(float, [Low_cutoff, High_cutoff, F_sample])
    
    #Convert cutoff frequencies into points on spectrum
    [Low_point, High_point] = map(lambda F: F/F_sample * M /2, [Low_cutoff, High_cutoff])# the division by 2 is because the spectrum is symmetric 

    Filtered_spectrum = [Spectrum[i] \
        if (i >= Low_point and i <= High_point) else 0.0 \
        for i in xrange(M)] # Filtering
    Filtered_signal = scipy.ifft(Filtered_spectrum, n=M)  # Construct filtered signal 
    return Spectrum, Filtered_spectrum, Filtered_signal, Low_point, High_point

from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 5000.0
    lowcut = 500.0
    highcut = 1250.0

#    # Plot the frequency response for a few different orders.
#    plt.figure(1)
#    plt.clf()
#    for order in [3, 6, 9]:
#        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#        w, h = freqz(b, a, worN=2000)
#        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
#
#    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
#             '--', label='sqrt(0.5)')
#    plt.xlabel('Frequency (Hz)')
#    plt.ylabel('Gain')
#    plt.grid(True)
#    plt.legend(loc='best')

    # Filter a noisy signal.
    T = 0.05
    nsamples = T * fs
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.show()
    
#    plt.figure(3)
#    plt.clf()
#    plt.plot(t, x, label='Noisy signal')
#    Spectrum, Filtered_spectrum, Filtered_signal, Low_point, High_point = bandpass_ifft(x, lowcut, highcut, fs, M=None)
#    plt.plot(t, Filtered_signal, 'r', label='Filtered signal (%g Hz)' % f0)
#    plt.xlabel('time (seconds)')
#    plt.hlines([-a, a], 0, T, linestyles='--')
#    plt.grid(True)
#    plt.axis('tight')
#    plt.legend(loc='upper left')
#
#    plt.show()