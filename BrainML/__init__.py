import tensorflow as tf
from BrainML.activation import Activator
from BrainML.layers import *
from BrainML.optimizer import Optimizer
from tensorflow.python.util import deprecation
##deprecation._PRINT_DEPRECATION_WARNINGS = False
##tf.compat.v1.disable_eager_execution()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Network: 
  def __init__(self,layers=None, name=None):
    self.model=None
    self.output=None
    self.layers=layers
    self.compile=None
    self.name=name
    newLayers=[]
    ##if layers[0].shape!=None:
    ##  newLayers.append(tf.keras.Input(input_shape=layers[0].shape))
    for i in range(0,len(layers)):
      newLayers.append(self.layers[i].layer)
      ##newLayers[i].value_to_feed=
    self.model=tf.keras.Sequential()##newLayers, name
    for i in newLayers:
      self.model.add(i)
    
    
  
  def train(self,x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
    validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
    sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_batch_size=None, validation_freq=1,
    max_queue_size=10, workers=1, use_multiprocessing=False,optimizer='rmsprop',
    loss=None, metrics=None, loss_weights=None,weighted_metrics=None, run_eagerly=None):
    if loss==None:
      loss="mse"
    elif metrics==None or metrics[0]=="all":
      metrics=["mae", "acc"]
    else:
      optimizer="rmsprop"
    
    self.compile=self.model.compile(optimizer, loss, metrics, loss_weights,weighted_metrics, run_eagerly)##initial_epoch,steps_per_epoch
    self.output=self.model.fit(x, y, batch_size, epochs, verbose, callbacks,validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, validation_steps, validation_batch_size, validation_freq,  max_queue_size, workers, use_multiprocessing)
    return self.output

  def Summary(self):
    self.model.summary()
  
##  if __name__ == "__main__":
##   pass