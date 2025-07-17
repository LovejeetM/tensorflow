import tensorflow as tf 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D 
from tensorflow.keras.layers import Dropout
import numpy as np 
import matplotlib.pyplot as plt 

input_layer = Input(shape=(28, 28, 1))

conv_layer = Conv2D(filters= 32, kernel_size= (3, 3), activation= 'relu', padding = 'same') (input_layer)

transpose_conv_layer = Conv2DTranspose(filters = 1, kernel_size = (3,3), activation = 'sigmoid', padding = 'same') (conv_layer)

model = Model(inputs = input_layer, outputs = transpose_conv_layer)

model.compile(optimizer = 'adam', loss= 'mean_squared_error')

X_train = np.random.rand(1000, 28, 28, 1)

y_train = X_train    # fro reconstruction

history = model.fit(X_train, y_train, epoches = 10, batch_size = 32, validation_split = 0.2)


X_test = np.random.rand(200, 28, 28, 1)

y_test = X_test

loss = model.evaluate(X_test, y_test)

y_pred = model.predict(X_test) 


# Plot
n = 10 

plt.figure(figsize=(20, 4))

for i in range(n): 

    # original 
    ax = plt.subplot(2, n, i + 1) 
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original") 
    plt.axis('off') 
    # reconstruction 
    ax = plt.subplot(2, n, i + 1 + n) 
    plt.imshow(y_pred[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.show() 




input_layer = Input(shape=(28, 28, 1))

conv_layer = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)
transpose_conv_layer = Conv2DTranspose(filters=1, kernel_size=(5, 5), activation='sigmoid', padding='same')(conv_layer)

model = Model(inputs=input_layer, outputs=transpose_conv_layer)

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')



# dropout layer
nput_layer = Input(shape=(28, 28, 1))


conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
dropout_layer = Dropout(0.5)(conv_layer)
transpose_conv_layer = Conv2DTranspose(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')(dropout_layer)

model = Model(inputs=input_layer, outputs=transpose_conv_layer)

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')


input_layer = Input(shape=(28, 28, 1))

conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='tanh', padding='same')(input_layer)
transpose_conv_layer = Conv2DTranspose(filters=1, kernel_size=(3, 3), activation='tanh', padding='same')(conv_layer)

model = Model(inputs=input_layer, outputs=transpose_conv_layer)

model.compile(optimizer='adam', loss='mean_squared_error')


history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

