import numpy as np


def time_windows(data, trig, fs, window=(-0.05, 0.6), aug=(0.02, 3)):
    y = []
    xs = []
    for i in np.where(trig != 0)[0]:
        if aug is not None:
            pert = np.linspace(-aug[0] * fs, aug[0] * fs, aug[1], dtype=int)
        else:
            pert = [0]
        for j in pert:
            xs.append(
                data[i + int(window[0] * fs) + j:i + int(window[1] * fs) + j])
            y.append((trig[i] + 1) / 2)
    y = np.array(y)
    X = np.stack(xs)
    exp_label = np.repeat(np.arange(0, 5), 240 * 3)
    return X, y, exp_label
