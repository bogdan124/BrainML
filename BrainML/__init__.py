import tensorflow as tf
from activation import Activator
from layers import Dense,Dropout
from optimizer import Optimizer



class Model:
  def __init__(self):
    self.xData=tf.compat.v1.placeholder("float64", name = "x")
    self.yData=tf.compat.v1.placeholder("float64", name = "y")
    self.Output=None
    self.layers=None
    self.trainning=False

  def Sequential(self,layers,name=None):##xData,yData,
      ##xData=tf.cast(xData,tf.float64) 
      ##yData=tf.cast(yData,tf.float64) 
      xData=self.xData
      yData=self.yData
      storeShape=None
      savePrevOutput=None
      saveNodesNumber=None
      self.layers=layers
      for i in range(0,len(layers)):
          if i==0:
              storeShape=layers[i].shapeData
              savePrevOutput=layers[i].calculate(storeShape[1],layers[i].numberOfNodes,xData)
              saveNodesNumber=layers[i].numberOfNodes
          elif i!=len(layers) or i!=0:
              if layers[i].type=="Dense":
                savePrevOutput=layers[i].calculate(saveNodesNumber,layers[i].numberOfNodes,savePrevOutput)
                saveNodesNumber=layers[i].numberOfNodes
              elif layers[i].type=="Dropout" and  self.trainning==True:
                print(savePrevOutput)
                savePrevOutput=layers[i].calculate(x=savePrevOutput)
                print(layers[i].output ,savePrevOutput)
          elif i==len(layers):
              savePrevOutput=layers[i].calculate(saveNodesNumber,layers[i].numberOfNodes,savePrevOutput)
      self.Output=layers[i].output        
      return layers[i].output 

  def compile(self,optimizer=None,loss=None):
      save=self.Output
      self.e = tf.reduce_mean(tf.math.squared_difference(self.yData, save))
      self.train = optimizer.minimize(self.e)

  def fit(self,x=None,y=None,batch_size=32,epochs=10,display=None):
      self.trainning=True
      save=self.Output
      self.epochs=epochs
      init = tf.compat.v1.global_variables_initializer()
      sess = tf.compat.v1.Session()
      sess.run(init)
      

      for i in range (self.epochs):
          error = sess.run(self.train, feed_dict={self.xData: x, self.yData: y})

          for j in display:
            if j=="epoch":
              print('\nEpoch: ' + str(i))
            elif j=="error":
              print('\nError: ' + str(sess.run(self.e, feed_dict={self.xData: x, self.yData: y})))
            elif j=="all":
               print('\nEpoch: ' + str(i)+'\tError: ' + str(sess.run(self.e, feed_dict={self.xData: x, self.yData: y})))

          if  i % self.epochs == 0:
              for el in sess.run(save, feed_dict={self.xData: x, self.yData: y}):
                  print('    ',el)
      sess.close() 