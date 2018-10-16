from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout,Reshape,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D,Conv1D,MaxPooling1D
from keras.models import model_from_json
import numpy as np
import os
import matplotlib.pyplot as plt
import xlrd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

#(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
train_data = np.loadtxt('train_data.txt',dtype='float', delimiter=',')
x_label = train_data[:,28]
train_data = np.delete(train_data, 28, 1)
train_data = np.delete(train_data, 27, 1)
train_data = np.delete(train_data, 0, 1)
input_shape = train_data.shape
#x_train = preprocessing.normalize(x_train)
# x_train = preprocessing.scale(train_data)

x_train, x_test, y_train, y_test = train_test_split(train_data, x_label, test_size = 0.2)

def build0():
    model=Sequential(name='speed')
    model.add(BatchNormalization(input_shape=(26,)))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

def build1():
    model=Sequential(name='speed')
    model.add(BatchNormalization(input_shape=(26,)))
    model.add(Dense(2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

def build2():
    model=Sequential(name='speed')
    model.add(BatchNormalization(input_shape=(26,)))
    model.add(Dense(13,activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

def build3():
    model=Sequential(name='speed')
    model.add(BatchNormalization(input_shape=(26,)))
    model.add(Dense(13,activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

def build4():
    model=Sequential(name='boston')
    model.add(BatchNormalization(input_shape=(26,)))
    model.add(Reshape((13,2)))
    model.add(Conv1D(filters=13,strides=1,padding='same',kernel_size=2,activation='sigmoid'))
    model.add(Conv1D(filters=26, strides=1, padding='same', kernel_size=2, activation='sigmoid'))
    model.add(MaxPooling1D(pool_size=2,strides=1,padding='same'))
    model.add(Conv1D(filters=52, strides=1, padding='same', kernel_size=2, activation='sigmoid'))
    model.add(Conv1D(filters=104, strides=1, padding='same', kernel_size=2, activation='sigmoid'))
    model.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

def build5():
    model = Sequential(name='speed')
    model.add(BatchNormalization(input_shape=(26,)))
    model.add(Reshape((13, 2,1)))
    model.add(Conv2D(filters=13, strides=1, padding='same', kernel_size=1, activation='sigmoid'))
    model.add(Conv2D(filters=26, strides=2, padding='same', kernel_size=2, activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=2, strides=1, padding='same'))
    model.add(Conv2D(filters=52, strides=1, padding='same', kernel_size=1, activation='sigmoid'))
    model.add(Conv2D(filters=104, strides=2, padding='same', kernel_size=2, activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=2, strides=1, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

for i in range(2):
    model=eval("build"+str(5)+"()")
    model.compile('adam','mae')#3.0895
    history=model.fit(x_train,y_train,batch_size=89,epochs=100,verbose=1,validation_data=(x_test,y_test))
    print(history.history)
   
    y_pred = model.predict(x_test)
    print(y_pred)
    f=open("result.txt",'a')
    f.write(str(history.history['val_loss'][-1])+"\n")
    f.close()

    fig = plt.figure(figsize=(20, 3))
    axes = fig.add_subplot(1, 1, 1)
    line3, = axes.plot(range(len(y_test)), y_test, 'g', label='actual')
    line1, = axes.plot(range(len(y_pred)), y_pred, 'b--', label='pred', linewidth=2)
    axes.grid()
    fig.tight_layout()
    plt.legend(handles=[line1, line3])
    plt.show()