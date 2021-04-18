from glob import glob
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report

from modules.data import *
from modules.plotting import *
from modules.classification import time_windows
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm import tqdm

if __name__ == '__main__':
    files = sorted(glob('./data/*.mat'))
    FS = 250
    for f in files:
        print(f)
        X, y = read_prepare(f)
        # All EEG
        # plot_signal(X, FS, True)
        # plt.plot(y)
        # plt.show()

        # Means to see if there is visible P300
        # Xt, yt, trial_n = time_windows(X,y,FS, aug=None)
        # plot_mean_std(Xt, yt)
        Xt, yt, trial_n = time_windows(X, y, FS)

        # Transform instead of shrinkage of LDA
        Xflat = Xt.reshape((Xt.shape[0], -1))
        pca = PCA(96)
        Xflat = pca.fit_transform(Xflat)
        print(f'PCA explained variance: {np.sum(pca.explained_variance_ratio_)}')

        i = 0
        preds = []
        for (trX, trY), (valX, valY) in kfold(Xflat, yt, trial_n):
            # print(f"LDA FOLD {i+1}")
            model = LinearDiscriminantAnalysis()
            lda = LinearDiscriminantAnalysis('svd')
            lda.fit(trX, trY)
            pred = lda.predict(valX)
            preds.append(pred)
            report = classification_report(valY[1::3], pred[1::3])
            # print(report)
        # print('==================')
        # print('FINAL KFOLD LDA')
        print('Leave trial out test')
        print(classification_report(yt, np.concatenate(preds)))


        print('FULL Dataset')
        lda = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)
        lda.fit(Xflat, yt)
        res = lda.transform(Xflat[1::3])
        print(f'All accuracy {lda.score(Xflat[1::3], yt[1::3])}')
        # Separability
        # plt.hist(res[yt[1::3] == 1], bins=100, alpha=0.5)
        # plt.hist(res[yt[1::3] == 0], bins=100, alpha=0.5)
        # plt.show()

        # Covariance of LDA
        # print(f'Overfit check - Coef mean: {lda.coef_.mean()}, Std: {lda.coef_.std()}')
        # plt.imshow(lda.covariance_)
        # plt.title('LDA covariance')
        # plt.show()

        cop = np.zeros_like(y)
        lda_f = np.zeros_like(y)
        before = int(0.05 * FS)
        p300 = int(0.6 * FS)
        for i in tqdm(range(len(y) - before - p300)):
            x = pca.transform(X[i:i + before + p300].reshape(1, -1))
            cop[i] = lda.predict(x)
            lda_f[i] = lda.transform(x)

        # Confusion matrix
        conf_matrix = np.zeros([2, 2])
        idxs = np.where(y != 0)[0]
        for i, v in enumerate(idxs):
            fnt = (v + idxs[i - 1]) // 2 if i > 0 and v - idxs[
                i - 1] < 100 else v - 25
            nxt = (v + idxs[i + 1]) // 2 if i < len(idxs) - 1 and idxs[
                i + 1] - v < 100 else v + 25
            pred = int(cop[fnt:nxt].max())
            conf_matrix[pred, (y[v] + 1) // 2] += 1
        sns.heatmap(conf_matrix / conf_matrix.sum(), annot=True,
                    xticklabels=['Non-Target', 'Target'],
                    yticklabels=['Non-Target', 'Target'])
        plt.show()
        print(f'Online TEST ACCURACY: {conf_matrix.diagonal().sum() / conf_matrix.sum()}')

        # Example of prediction
        # plt.plot(y[10000:11000], label='Trigger')
        # plt.plot(cop[10000:11000], label='Predicted')
        # plt.plot(lda_f[10000:11000], label='LDA Score')
        # plt.show()
        print('=============================')
