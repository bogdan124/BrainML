import tensorflow as tf



class Conv2D:
  def __init__(self, 
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               groups=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None
               ):
    self.layer=tf.keras.layers.Conv2D(
                filters,
                kernel_size,
                strides,
                padding,
                data_format,
                dilation_rate,
                activation,
                use_bias,
                kernel_initializer,
                bias_initializer,
                kernel_regularizer,
                bias_regularizer,
                activity_regularizer,
                kernel_constraint,
                bias_constraint)


class InputLayer:
  def __init__(self,shape):
      self.layer=tf.keras.Input(shape=shape)


class Flatten:
  def __init__(self):
      self.layer=tf.keras.layers.Flatten()


class MaxPolling2D:
  def __init__(self,pool_size):
      self.layer=tf.keras.layers.MaxPooling2D(pool_size=pool_size)


class Dropout:
  def __init__(self,value):
    self.layer=tf.keras.layers.Dropout(value)


class fully_connected:
  def __init__(self, units, activation=None,shape=None, use_bias=True, kernel_initializer='glorot_uniform',bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,activity_regularizer=None, kernel_constraint=None, bias_constraint=None,value_to_feed=None):
    '''if shape!=None:
      self.shape=shape
    else:
      self.shape=None
    self.value_to_feed=value_to_feed
    if self.value_to_feed!=None
      self.layer=layers.Dense(units, activation, use_bias, kernel_initializer, bias_initializer,kernel_regularizer, bias_regularizer,activity_regularizer, kernel_constraint, bias_constraint)(value_to_feed)
    else:'''
    self.layer=tf.keras.layers.Dense(units, activation, use_bias, kernel_initializer, bias_initializer,kernel_regularizer, bias_regularizer,activity_regularizer, kernel_constraint, bias_constraint)
    