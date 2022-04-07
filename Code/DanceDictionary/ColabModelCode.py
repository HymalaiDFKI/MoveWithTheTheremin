####################################################################
# 1D conv Model 
####################################################################
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
print('Numpy' + np.__version__)
print('TensorFlow' + tf.__version__)
print('Keras' + tf.keras.__version__)

###################### create model

from tensorflow import keras
from tensorflow.keras import layers
def build_Conv1D(filters = 40, kernel = (10), dense=100, numClass = numClass):
  model = keras.models.Sequential([
        layers.Reshape((400,4),input_shape = (400,4,1)),
        layers.Conv1D(filters= filters, kernel_size=kernel, padding='same', activation='relu',input_shape=(400,4)),
        layers.MaxPooling1D(pool_size=(10)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Conv1D(filters= filters, kernel_size= kernel, padding= 'same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=(10)),
        layers.Dropout(0.2),
        layers.Conv1D(filters= filters, kernel_size = kernel, padding='same', activation='relu'), 
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(dense, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(numClass, activation='softmax')                       
  ])
  return model

#################### view model
model = build_Conv1D()
model.summary()

################### optimizer
opt = keras.optimizers.SGD(learning_rate=0.005, momentum=0.001)
model.compile(optimizer= opt, loss= keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

################### train model
epoch = 500
history = model.fit(x_train,y_onehot_train, epochs=epoch)