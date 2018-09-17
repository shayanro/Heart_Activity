import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM#, CuDNNLSTM
import keras
import pickle
import os
import numpy as np
from tensorflow import reset_default_graph
reset_default_graph()
path = 'D:\PAMAP_Dataset'
os.chdir(path)

f = open('X.pickle', 'rb')
X = pickle.load(f)
f = open('y.pickle', 'rb')
y = pickle.load(f)

T = int(0.1 * 60)  # 5 minutes
data_len = 3*T*50 + 1  # 50 for 50Hz

X = np.array(X).reshape(-1, data_len, 1)
y = np.array(y).reshape(-1, 1)

model = Sequential()

# IF you are running with a GPU, try out the CuDNNLSTM layer type instead (don't pass an activation, tanh is required)

model.add(CuDNNLSTM(512, input_shape=(X.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, kernel_initializer='normal'))

opt = tf.keras.optimizers.Adam(lr=0.0005, decay=1e-6)
# Compile model
model.compile(
    loss='mse',
    optimizer=opt,
    metrics=['mse'])


model.fit(X, y,
          batch_size=64,
          epochs=5,
          validation_split=0.05)
