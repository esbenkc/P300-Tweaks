import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


def plot_signal(signals, fs=250, sep=False):
    plt.figure(figsize=[20, 10])
    N = signals.shape[0]
    for i in range(signals.shape[1]):
        if sep:
            plt.subplot(8, 1, i + 1)
        plt.plot(np.linspace(0, N / fs, N), signals[:, i])
    plt.show()


def plot_spectrum(signal, fs, title=None):
    f, Pxx_den = sig.periodogram(signal, fs)
    plt.semilogy(f, Pxx_den)
    plt.ylim([1e-8, 1e3])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    if title is not None:
        plt.title(title)
    plt.show()


def plot_mean_std(X, y):
    plt.figure(figsize=(20, 10))
    t = np.linspace(-50, 600, X.shape[1])
    for i in range(8):
        plt.subplot(4, 2, i + 1)
        mean = X[y == 1, :, i].mean(0)
        std = X[y == 1, :, i].std(0)
        plt.plot(t, mean, label='Trigger')
        plt.fill_between(t, mean - std, mean + std, alpha=0.3)
        mean = X[y == 0, :, i].mean(0)
        std = X[y == 0, :, i].std(0)
        plt.plot(t, mean, label='Non-Trigger')
        plt.fill_between(t, mean - std, mean + std, alpha=0.3)
        plt.legend()
        plt.xlabel('Time relative to trigger: ms')
    plt.show()
