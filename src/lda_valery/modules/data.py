from mne.externals.pymatreader import read_mat
import scipy.signal as sig
import numpy as np


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sig.lfilter(b, a, data)
    return y


def read_prepare(filename):
    raw = read_mat(filename)
    fs = 250
    trig = raw['trig']
    data = raw['y']
    filtered = data.copy()
    for i in range(filtered.shape[1]):
        filtered[:, i] = butter_bandpass_filter(data[:, i], 0.5, 30, fs)
        filtered[:, i] = np.convolve(filtered[:, i], 7, 'same')
    filtered = (filtered - filtered.mean(0)[None]) / filtered.std(0)[None]
    # pca = PCA(8)
    # filtered = pca.fit_transform(filtered)
    # print(pca.explained_variance_ratio_)
    return filtered, trig


def kfold(X, y, trial_labels):
    for t in np.unique(trial_labels):
        trX = X[np.where(trial_labels != t)]
        valX = X[np.where(trial_labels == t)]
        trY = y[np.where(trial_labels != t)]
        valY = y[np.where(trial_labels == t)]
        yield (trX, trY), (valX, valY)
