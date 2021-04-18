import mne
from mne.io.kit.kit import _make_stim_channel
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
    # X_selected = X_samples
    
    events = []
    for i, id in enumerate(flash):
        if id != 0:
            events.append(np.array([i,0,id]))

    events = np.array(events)
    events = events.astype(int)

    # Get all the right IDs
    # falseX = X_selected[false_idx]
    # falsey = y[false_idx]

    finalX  = X_samples
    # truey  = y[true_idx]
    # proportional data to avoid greedy cost funtion

    # proportionalX = falseX[:int(len(trueX))]
    # proportionaly = falsey[:int(len(truey))]

    # finaly = np.concatenate((truey, proportionaly))

    X_timeseries = np.vstack(finalX)
    X_letters = X_timeseries.reshape(len(flash_active),8,351)
    # y_letters = finaly.reshape(15,20,2)
    # cleaned_Y = np.vstack(y_letters)

    return X_letters, events

fs = 250

n_channels = 8
sampling_freq = 250
ch_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
ch_types = ['eeg'] * 8

info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)


for subject in range(1,6):
    # shape = (n_epochs, n_channels, n_steps)
    data = si.loadmat('C:\\Users\\esben\\Desktop\\BCI\\hackthat-p300\\data\\S{0}.mat'.format(subject))
    data, events = clean_data(data['y'], data['trig'])

    # events shape = (n_events, 3)
    events_dict = {'no-target': -1, 'target': 1}
    epochs = mne.EpochsArray(data, info, events = events, event_id = events_dict)

    epochs.save('C:\\Users\\esben\\Desktop\\BCI\\hackthat-p300\\data\\S{0}.fif'.format(subject), overwrite=True)

