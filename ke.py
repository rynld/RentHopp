# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import Sequential
import numpy as np
import pandas as pd
from RentHop import RentHop
from sklearn.model_selection import train_test_split

data_dim = 31
nb_classes = 3

train_df = pd.read_json("input/train.json")
train_df = train_df.head(200)
test_df = pd.read_json("input/test.json")
print("Read files")
rhop = RentHop()

train_X, train_Y = rhop.getTrainNet(train_df)
test_X = rhop.getTest(test_df)

x_train, x_test, y_train, y_test = train_test_split(train_X,train_Y,test_size=0.3)


model = Sequential()
model.add(Dense(500, input_dim=data_dim, init='uniform'))
model.add(Activation('tanh'))
#model.add(Dropout(0.5))
# model.add(Dense(64, init='uniform'))
# model.add(Activation('tanh'))
#model.add(Dropout(0.5))
model.add(Dense(nb_classes, init='uniform'))
model.add(Activation('softmax'))
#
model.compile(loss='categorical_crossentropy',
           optimizer='sgd')

model.fit(x_train, y_train,
          nb_epoch=20,
          batch_size=16)

score = model.evaluate(x_test, y_test, batch_size=16)
print("Score :" + score)
