import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def prepare_model():
  input_layer = keras.Input(shape = (351,8,1), name='main_input')
  x     = layers.Conv2D(16, 8, padding='same', activation='relu')(input_layer)
  x     = layers.Dropout(0.33)(x)
  x     = layers.Conv2D(32, 6, padding='same', activation='relu')(x)
  x     = layers.Conv2D(8, 4, padding='same', activation='relu')(x)
  x     = layers.Conv2D(4, 2, padding='same', activation='relu')(x)
  x     = layers.GlobalAveragePooling2D()(x)
  x     = layers.Dense(8)(x)
  x     = layers.Dense(64)(x)
  output = layers.Dense(2, activation='softmax')(x)

  model = keras.Model(inputs=input_layer, outputs=output)

  return model