import mne
import numpy as np
import scipy.io as si
from keras.utils import to_categorical

def clean_data(X, flash):
    # in kaggle: flash 0 ´= sample start, flash1 = duration, flash2 = stimulation, flash3 = hit/nohit
    flash_active = [(i, n[0]>0) for (i, n) in enumerate(flash) if n[0] != 0]

    X_samples = np.array([np.array(X[i[0]:i[0]+351]) for i in flash_active] )

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
    X_letters = X_timeseries.reshape(15,20,8,351)
    y_letters = finaly.reshape(15,20,2)
    cleaned_X = np.vstack(X_letters)
    cleaned_Y = np.vstack(y_letters)

    return cleaned_X, cleaned_Y

fs = 250

n_channels = 8
sampling_freq = 250
ch_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
ch_types = ['eeg'] * 8

info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)


for subject in range(1,6):

    # shape = (n_epochs, n_channels, n_steps)
    data = si.loadmat('..\\data\\S{0}.mat'.format(subject))

    data, y = clean_data(data['y'], data['trig'])
    epochs = mne.EpochsArray(data, info)

    epochs.save('..\\data\\S{0}.fif'.format(subject))

