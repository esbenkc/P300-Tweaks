from scipy.signal import butter, lfilter
from mat4py import loadmat

def butter_bandpass(cutoff = 5, fs = 250.0, order = 6):
    nyq = 0.5 * fs
    if not isinstance(cutoff, list):
        cutoff = cutoff / nyq
    else:
        cutoff = [cutoff[0] / nyq, cutoff[1] / nyq]
    b, a = butter(order, cutoff, btype='band', analog=False)
    return b, a

def butter_bandpass_filter(data, cutoff = 5, fs = 250.0, order = 6):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def normalized(vec):
  norm_vec = (vec - vec.min(axis=1, keepdims=True))/vec.ptp(axis=1, keepdims=True)
  return norm_vec

def prepare_data(file, cutoff = 5, fs = 250.0, order = 6):
  raw_data = loadmat(file)
  useful_data = raw_data.copy()
  X = useful_data['y']
  Y = useful_data['trig']
  X_filtered = butter_bandpass(X, cutoff, fs, order)
  
  return X, X_filtered, Y


