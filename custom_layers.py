import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Softmax
import numpy as np 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dropout


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




x_train = np.random.random((1000, 20)) 
y_train = np.random.randint(10, size=(1000, 1)) 

# categorical one-hot encoding 
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10) 
model.fit(x_train, y_train, epochs=10, batch_size=32) 



x_test = np.random.random((200, 20)) 
y_test = np.random.randint(10, size=(200, 1)) 

# labels to categorical one-hot encoding 
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10) 

loss = model.evaluate(x_test, y_test) 
print(f'Test loss: {loss}') 



# for visualizing the model architecture
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)



model = Sequential([CustomDenseLayer(256), Dropout(0.9), CustomDenseLayer(10)])
model.compile(optimizer= 'adam', loss = "categorical_crossentropy")
model.fit(x_train, y_train, epochs = 10, batch_size = 32)


class CustomDenseLayer(Layer):
    def __init__(self, units=128):
        super(CustomDenseLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)

model = Sequential([
    CustomDenseLayer(128),
    CustomDenseLayer(10)
])

model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit(x_train, y_train, epochs=10, batch_size=32)
