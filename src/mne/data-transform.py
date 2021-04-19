import mne
from mne.io.kit.kit import _make_stim_channel
import numpy as np
import scipy.io as si

def clean_data(X, flash, output="raw"):
    
    if output == "epochs":
        flash_active = [(i, n[0]>0) for (i, n) in enumerate(flash) if n[0] != 0]

        X_samples = np.array([np.transpose(np.array(X[i[0]:i[0]+351])) for i in flash_active] )

        events = []
        for i, id in enumerate(flash):
            if id != 0:
                events.append(np.array([i,0,id]))

        events = np.array(events)
        events = events.astype(int)

        return X_samples, events


def data_transformation_epochs(path):
    sampling_freq = 250
    ch_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
    ch_types = ['eeg'] * 8

    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)


    for subject in range(1,6):
        # shape = (n_epochs, n_channels, n_steps)
        data = si.loadmat((path + 'S{0}.mat').format(subject))
        data, events = clean_data(data['y'], data['trig'], output="epochs")

        # events shape = (n_events, 3)
        events_dict = {'no-target': -1, 'target': 1}
        epochs = mne.EpochsArray(data, info, events = events, event_id = events_dict)

        epochs.save((path + 'S{0}.fif').format(subject), overwrite=True)

    return("success")

def data_transformation_raw(path):
    sampling_freq = 250
    ch_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8','MNE_STIM_CHANNEL']
    ch_types = ['eeg'] * 8 + ['stim']

    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sampling_freq)
    
    for subject in range(1,6):
        # shape = (n_channels, n_steps)
        data = si.loadmat((path + 'S{0}.mat').format(subject))
        data, events = (data['y'], data['trig'])
        
        ev = []
        for idx, e in enumerate(events):
            if e == -1: 
                ev.append(2)
            else: ev.append(e)

        events = np.array(ev)    
        data = np.c_[data, events]
        
        data = np.transpose(data)

        raw = mne.io.RawArray(data, info)
        montage = mne.channels.make_standard_montage(kind="standard_1020")
        raw.set_montage(montage)
        
        raw.save((path + 'R_S{0}.fif').format(subject), overwrite=True)

    return("success")

data_transformation_raw("C:\\Users\\esben\\Desktop\\BCI\\hackthat-p300\\data\\")
# data_transformation("C:\\Users\\esben\\Desktop\\BCI\\hackthat-p300\\data\\")


