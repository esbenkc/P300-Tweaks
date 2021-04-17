from scipy.signal import butter, lfilter
from mat4py import loadmat

def butter_lowpass(cutoff = 5, fs = 256.0, order = 6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff = 5, fs = 256.0, order = 6):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def normalized(vec):
  norm_vec = (vec - vec.min(axis=1, keepdims=True))/vec.ptp(axis=1, keepdims=True)
  return norm_vec

def prepare_data(file, cutoff = 5, fs = 256.0, order = 6):
  raw_data = loadmat(file)
  useful_data = raw_data.copy()
  X = useful_data['y']
  Y = useful_data['trig']
  X_filtered = butter_lowpass_filter(X, cutoff, fs, order)
  
  return X, X_filtered, Y


