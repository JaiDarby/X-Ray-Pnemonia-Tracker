import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
import pickle

from tensorflow.python.keras.layers.pooling import MaxPooling2D

x = pickle.load(open("DatasetX.pickle", "rb"))
y = pickle.load(open("DatasetY.pickle", "rb"))

x = x/255.0

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x, y, batch_size=35, epochs=10, validation_split=0.1)

print("Model Trained")