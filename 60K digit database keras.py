import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input
from keras.datasets import mnist
(x_try,y_try),(x_tst,y_tst)=mnist.load_data()
x_try.shape
#plt.imshow(x_try[0])
x_try.shape
x=x_try.reshape((60000,784))
y_try.shape
y_try[0:5]
one_hot_y=keras.utils.to_categorical(y_try)
one_hot_y.shape
one_hot_y[0]
x.shape
one_hot_y.shape
inp=Input(shape=(784,))
hid=Dense(10,activation='sigmoid')(inp)
out=Dense(10,activation='sigmoid')(hid)
model=Model(inp,out)
model.compile(optimizer='SGD',loss='MSE',metrics=['accuracy'])  ##accuracy-kitni hai 1 eboch k bad
model.fit(x,one_hot_y)
model.fit(x,one_hot_y,epochs=5 )
x_t=x_tst.reshape((10000,784))
pred=model.predict(x_t)
c=0
for i in range(0,len(y_tst)):
  if(np.argmax(pred[i])==y_tst[i]):
    c+=1    
print (c)
print ("Accuracy are:- ")
print ((float(c)*100.00)/len(y_tst))