import tensorflow as tf


class Optimizer:

  def __init__(self,optimizerName=None):
    """
    Here you have the Optimizer class, optimizerName is the optimizer that you want to chose:
    ex: Optimizer("Adam")
    optimizerName can be one of the following Adam, Adadelta, Adagrad, Adamax, Nadam, Sgd, Rmsprop
    """
    self.optimzerChosen=None
    self.value=None
    self.optimizers={"adam":[1,Adam()],"adadelta":[2,Adadelta()],"adagrad":[3,Adagrad()],"adamax":[4,Adamax],
                     "nadam":[5,Nadam()],"sgd":[6,Sgd()],"rmsprop":[7,Rmsprop()]}

    if optimizerName is not None:
      optimizerName=str(optimizerName)
      optimizerName=optimizerName.lower()
      print(optimizerName)
      self.optimzerChosen=self.optimizers[str(optimizerName)][1]
      self.value=self.optimzerChosen
    
def Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False,name='Adam'):
    """
    Here you have the Adam Optimizer
    """
    adam=tf.compat.v1.train.AdamOptimizer(learning_rate, beta_1, beta_2, epsilon, amsgrad,name )
    return adam

def Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name='Adadelta'):
    """
    Here you have the Adadelta Optimizer
    """
    adadelta=tf.compat.v1.train.AdadeltaOptimizer(learning_rate, rho, epsilon, name)
    return adadelta

def Adamax():
    """
    Here you have the Adamax Optimizer
    """
    pass
  
def Adagrad():
    """
    Here you have the Adagrad Optimizer
    """
    pass
  
def Nadam():
    """
    Here you have the Nadam Optimizer
    """
    pass

def Sgd(learning_rate=0.001):
    """
    Here you have the Stochastic Gradient Descent Optimizer
    """
    sgd=tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    return sgd

def Rmsprop():
    """
    Here you have the Rmsprop Optimizer
    """
    pass
