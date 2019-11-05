import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

# import dataset from keras
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# check the shape and size
x_train.shape
x_test.shape

# check the visual of index
plt.matshow(x_train[0])

# check the index label
y_train[0]

# Normalization
x_train = x_train/255
x_test = x_test/255

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

# start the models
model = Sequential()

# Input layer
model.add(Flatten(input_shape=[28,28]))

# Hidden layer
model.add(Dense(20,activation='relu'))

# output layer
model.add(Dense(10,activation='softmax'))

# summary of model
model.summary()

# compile 
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# fit the models
model.fit(x_train, y_train, epochs = 5)

# check the prediction
plt.matshow(x_test[0])
 
x_test.shape

# prediction
yp = model.predict(x_test)

# check the prediction
yp[0]

np.argmax(yp[0])

# check the accuracy of trained model
model.evaluate(x_test, y_test)



#+++++++SAVE THE MODEL+++++++++
#this model useful for when we have lots of array

from sklearn.externals import joblib

#save
joblib.dump(model,'model_joblib')

#retrive the model
ab = joblib.load('model_joblib')

# use agian the model
ab.predict(x_test)




