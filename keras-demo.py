from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import numpy as np
from keras.utils import np_utils
from keras.datasets import mnist
seed = 1 #for repreducable random results
np.random.seed(seed)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28) #resize the data so it works with the network
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
#covert data to float and normalize it for processing
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1, 28, 28))) #base input layer: takes a 28 by 28 image of depth 1
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) # 2 by 2 filter taking the max of 4 values
model.add(Dropout(0.25)) #prevents overfitting
model.add(Flatten()) #must be flattened before dense layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) #simulates 1-10 output for mnist digit probability
model.compile(loss='categotrical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, nb_epoch=10, verbose=1)
score = model.evaluate(x_test, y_test, verbose=0)