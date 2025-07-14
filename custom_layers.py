import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Softmax


# Creating a custom layer to use in our neural network

class CustomDenseLayer(Layer):
    def __init__(self, units = 32):
        super(CustomDenseLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape = (input_shape[-1], self.units), initializer='random_normal', trainable = True)
        
        self.b = self.add_weight(shape = (self.units,), initializer = 'zeros', trainable = True)
    
    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)

model = Sequential([CustomDenseLayer(128), CustomDenseLayer(10),  Softmax()])

model.compile(optimizer='adam', loss='categorical_crossentropy')
print("Model summary before building:")
model.summary()


model.build((1000, 20))
print("\nModel summary after building:")
model.summary()