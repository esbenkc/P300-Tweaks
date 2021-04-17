# from modules import filter
import glob
# import numpy as np
import matplotlib.pyplot as plt 

data_dir = './data/*.mat'
files = glob.glob(data_dir)

# Get the filter coefficients so we can check its frequency response.
# b, a = filter.butter_lowpass(cutoff = [0.5, 30.0], fs = 24.0)


# for file in files:
#     X_Filtered, flash = filter.prepare_data(file, cutoff = [0.5, 30], fs = 250.0)
#     # Filtered signal
#     array8D = np.array(X_Filtered)
#     for i in range(0, 8):
#       plt.plot(array8D[:, i], label = 'Channel: ' + str(i))
#     plt.plot(flash, label='Trigger')
#     plt.legend()
#     plt.show()
#     exit()

from modules import modeling
model = modeling.model_prepare()
modeling.model_compile(model)

acc, val_acc, loss, val_loss = modeling.train_net(model, files)

plt.rcParams["figure.figsize"] = (10,7)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()