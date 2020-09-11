import tensorflow as tf

class Dense:

    def __init__(self,numberOfNodes,shape=None,activation=None,x_data=None):
        self.numberOfNodes=numberOfNodes
        self.activation=activation 
        self.shapeData=shape
        self.type="Dense"
        self.output=None

    def calculate(self,dim1,dim2,dataCalculate):
        self.weights = tf.Variable(tf.cast(tf.random.normal([dim1,dim2]),tf.float64),dtype='float64')
        self.biases  =  tf.Variable(tf.cast(tf.random.normal([dim2]),tf.float64),dtype='float64')
        self.output=tf.keras.activations.sigmoid(tf.add(tf.matmul(dataCalculate, self.weights), self.biases))
        return self.output

class Dropout:

    def __init__(self,rate,x=None,noise_shape=None,seed=1, name=None):
        self.rate=rate
        self.noise_shape=noise_shape
        self.seed=seed
        self.name=name
        self.type="Dropout"
        self.output=None
    
    def calculate(self,x):
        value=tf.nn.dropout(x, self.rate, noise_shape=self.noise_shape, seed=self.seed, name=self.name)
        self.output=value


##not sure here may be come back later
class Flatten:

    def __init__(self):
        self.flat=None
    
    def calculate(self,x):
        self.flat=tf.reshape(x,[-1, x.get_shape().as_list()[0]])