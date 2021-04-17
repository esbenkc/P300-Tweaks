import glob
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt 
from mat4py import loadmat

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

data_dir = './data/*.mat'
files = glob.glob(data_dir)

# Filter requirements.
order = 6
fs = 256.0       # sample rate, Hz
cutoff = 1.5  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

def prepare_data(file):
  raw_data = loadmat(file)
  useful_data = raw_data.copy()
  X = useful_data['y']
  Y = useful_data['trig']
  X_filtered = butter_lowpass_filter(X, cutoff, fs, order)
  
  return X, X_filtered, Y


for file in files:
    X, X_Filtered, Y = prepare_data(file)
    # Filtered signal
    array8D = np.array(X_Filtered)
    for i in range(0, 7):
      plt.plot(array8D[:, i])
    plt.show()