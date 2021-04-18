import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from modules import prefilter
import numpy as np
import time


def model_prepare():
  # try different values for shape parameter 1
  # shape = (time, channels, ur mom)
  input_layer = keras.Input(shape = (351,1,1), name='main_input')
  x     = layers.Conv2D(16, 1, padding='same', activation='relu')(input_layer)
  x     = layers.Dropout(0.33)(x)
  x     = layers.Conv2D(32, 1, padding='same', activation='relu')(x)
  x     = layers.Conv2D(8, 1, padding='same', activation='relu')(x)
  x     = layers.Conv2D(4, 1, padding='same', activation='relu')(x)
  x     = layers.GlobalAveragePooling2D()(x)
  x     = layers.Dense(8)(x)
  x     = layers.Dense(64)(x)
  output = layers.Dense(2, activation='softmax')(x)

  model = keras.Model(inputs=input_layer, outputs=output)

  return model

def model_compile(model):
  #compiling the model
  #opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)  # default params
  opt = keras.optimizers.Nadam()
  model.compile(optimizer=opt,
    loss='binary_crossentropy',
    metrics=['accuracy'])

def train_net(model, files):
  appX = []
  appy = []
  init = time.time()
  for file in files:
    X, flash  = prefilter.prepare_data(file, cutoff = [0.5, 30], fs = 250.0)
    X_clean, y_clean = prefilter.clean_data(X, flash)
    appX.append(X_clean)
    appy.append(y_clean)
  X = [subject for subject in appX]
  y = [subject for subject in appy]
  X_train, X_valid, y_train, y_valid = train_test_split(np.vstack(X), np.vstack(y), test_size=0.1)
  history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=50, epochs=150, verbose=1)
  end = time.time()
  print("time elapsed training is:", (end - init)/60, " minutes")  
  return history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss']

def train_net_intra(model, file):
  appX = []
  appy = []
  init = time.time()
  X, flash  = prefilter.prepare_data(file, cutoff = [0.5, 30], fs = 250.0)
  X_clean, y_clean = prefilter.clean_data(X, flash)
  appX.append(X_clean)
  appy.append(y_clean)
  X = [subject for subject in appX]
  y = [subject for subject in appy]
  X_train, X_valid, y_train, y_valid = train_test_split(np.vstack(X), np.vstack(y), test_size=0.1)
  history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=10, epochs=50, verbose=1)
  end = time.time()
  print("time elapsed training is:", (end - init)/60, " minutes")  
  return history.history['accuracy'], history.history['val_accuracy'], history.history['loss'], history.history['val_loss']