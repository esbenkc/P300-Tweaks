# from modules import filter
import glob
import time
# import numpy as np
import matplotlib.pyplot as plt
import numpy as np

data_dir = './data/*.mat'
files = glob.glob(data_dir)

from modules import modeling, prefilter

# for file in files:
#     X_Filtered, flash = prefilter.prepare_data(file, cutoff = [0.5, 30], fs = 250.0)
#     # Filtered signal
#     array8D = np.array(X_Filtered)
#     for i in range(0, 8):
#       plt.plot(array8D[:, i], label = 'Channel: ' + str(i))
#     plt.plot(flash, label='Trigger')
#     plt.legend()
#     plt.show()
#     exit()




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




# Intra participant
# for file in files:
#   model = modeling.model_prepare()
#   modeling.model_compile(model)
#   acc, val_acc, loss, val_loss = modeling.train_net_intra(model, file)
#   plt.rcParams["figure.figsize"] = (10,7)
#   plt.plot(acc)
#   plt.plot(val_acc)
#   plt.title('Model accuracy')
#   plt.ylabel('Accuracy')
#   plt.xlabel('Epoch')
#   plt.legend(['Train', 'Valid'], loc='upper left')
#   plt.show()


test_subject = files.pop(1)
test_subject_2 = files.pop(2)

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



from sklearn.model_selection import train_test_split

X, Flash = prefilter.prepare_data(test_subject, cutoff = [0.5, 30], fs = 250.0)
X_clean, y_clean = prefilter.clean_data(X, Flash)
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.1)
history = model.fit(X_train, y_train, batch_size=1, epochs=30)
 
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Test'], loc='upper left')
plt.show()

# Plot test loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test'], loc='upper left')
plt.show()

score = model.evaluate(X_test, y_test)

init = time.time()
preds = model.predict(X_test)
end = time.time()
print("time elapsed for each trial is:",(end - init)/X_test.shape[0] * 1000, "ms")

from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test.argmax(axis=1), preds.argmax(axis=1))


matrix_norm = np.zeros((2,2))
for i in range(2):
  matrix_norm[i] = matrix[i]/matrix[i].sum(axis=0)

import seaborn as sns
import pandas as pd
df_cm = pd.DataFrame(matrix_norm, columns=np.unique(y_test), index = np.unique(y_test))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size