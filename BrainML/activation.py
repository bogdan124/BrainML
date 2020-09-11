import tensorflow as tf


class Activator:
  def __init__(self,activationFunc):
    self.function=None
    self.activationFunc=activationFunc
    self.optimizers={"sigmoid":[1,sigmoid()],"relu":[2,relu()],
                     "elu":[3,elu()],"softmax":[4,softmax()],
                     "exponential":[5,exponential()],"elu":[6,elu()],"tanh":[7,tanh()]}
                     
    if type(activationFunc,str)==True:
        self.function=self.optimizers[str(activationFunc)][1]

def sigmoid(x=None):
  return tf.keras.activations.sigmoid(x)


def relu(x=None, alpha=0.0, max_value=None, threshold=0):
  return tf.keras.activations.relu(x, alpha, max_value, threshold)


def elu(x=None, alpha=1.0):
  return tf.keras.activations.elu(x,alpha)


def tanh(x=None):
  return tf.keras.activations.tanh(x)


def exponential(x=None):
  return tf.keras.activations.exponential(x)


def softmax(x=None,axis=-1):
  return tf.keras.activations.softmax(x, axis)


##activ=Activator("sigmoid")
##print(activ.function)