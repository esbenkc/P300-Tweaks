from modules import filter
import glob
import numpy as np
import matplotlib.pyplot as plt 

data_dir = './data/*.mat'
files = glob.glob(data_dir)

# Get the filter coefficients so we can check its frequency response.
b, a = filter.butter_lowpass()


for file in files:
    X, X_Filtered, Y = filter.prepare_data(file)
    # Filtered signal
    array8D = np.array(X_Filtered)
    for i in range(0, 7):
      plt.plot(array8D[:, i])
    plt.show()