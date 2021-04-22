from scipy.signal import butter, lfilter
from mat4py import loadmat
import numpy as np
from keras.utils import to_categorical

def butter_bandpass(cutoff = 5, fs = 250.0, order = 6):
    nyq = 0.5 * fs
    if not isinstance(cutoff, list):
        cutoff = cutoff / nyq
    else:
        cutoff = [cutoff[0] / nyq, cutoff[1] / nyq]
    b, a = butter(order, cutoff, btype='band', analog=False)
    return b, a

def butter_bandpass_filter(data, cutoff = 5, fs = 250.0, order = 6):
    b, a = butter_bandpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def normalized(vec):
  norm_vec = (vec - vec.min(axis=1, keepdims=True))/vec.ptp(axis=1, keepdims=True)
  return norm_vec

def prepare_data(file, cutoff = 5, fs = 250.0, order = 6):
  raw_data = loadmat(file)
  useful_data = raw_data.copy()
  array8D = np.array(useful_data['y'])
  X = np.delete(array8D, 3, 1)
  #X = useful_data['y']
  flash = useful_data['trig']
  X_filtered = butter_bandpass_filter(X, cutoff, fs, order)
  
  return X_filtered, flash


def clean_data(X, flash):
    # in kaggle: flash 0 ´= sample start, flash1 = duration, flash2 = stimulation, flash3 = hit/nohit
    flash_active = [(i, n[0]>0) for (i, n) in enumerate(flash) if n[0] != 0]

    X_samples = np.array([np.array(X[i[0]-25:i[0]+275]) for i in flash_active] )

    # X_samples = np.array([np.array(X[i[0]:i[0]+351]) for i in flash] )
    #label     = [i[3] - 1 for i in flash]
    label     = [i[1] for i in flash_active]

    # LIMIT = 4080 #the last trial is incomplete
    # X_selected = np.array(X_samples[:LIMIT])
    X_selected = X_samples
    # label_selected = np.array(label[:LIMIT])
    label_selected = label
    y = np.array(to_categorical(label_selected))
    false_idx = [k for k, i in enumerate(y) if i[0] == 1]
    true_idx  = [k for k, i in enumerate(y) if i[0] == 0]

    falseX = X_selected[false_idx]
    falsey = y[false_idx]

    trueX  = X_selected[true_idx]  
    truey  = y[true_idx]
    # proportional data to avoid greedy cost funtion

    proportionalX = falseX[:int(len(trueX))]
    proportionaly = falsey[:int(len(truey))]

    finalX = np.concatenate((trueX, proportionalX))
    finaly = np.concatenate((truey, proportionaly))

    X_timeseries = np.vstack(finalX)
    X_letters = X_timeseries.reshape(20,15,300,7)
    y_letters = finaly.reshape(20,15,2)
    cleaned_X = np.vstack(X_letters)
    cleaned_Y = np.vstack(y_letters)

    return cleaned_X, cleaned_Y