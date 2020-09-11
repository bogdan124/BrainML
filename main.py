import tensorflow as tf
import DeepML as dl
from DeepML.layers import Dense,Dropout
from DeepML.optimizer import Optimizer,Adam
##from DeepML.activation import Activator,relu,softmax

model=dl.Model()
save=model.Sequential([
                 Dense(4,activation="relu",shape=[2,2]),
                 Dense(8, activation="relu"),
                 Dense(2, activation="relu"),
                 Dropout(0.9),
                 Dense(1, activation="relu")
                 ]
                ,name="model1")

X = [[0,0],[0,1],[1,0],[1,1]]
Y = [[0],[1],[1],[0]]

##print(dir(tf))
##optim=Optimizer("adam")
optim=Adam(learning_rate=0.01)
model.compile(optimizer=optim)
model.fit(X,Y,epochs=100,display=["all"])##all
