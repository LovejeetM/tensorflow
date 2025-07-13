from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dense 
import warnings
import numpy as np 
from tensorflow.keras.layers import Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Dense, Input
from tensorflow.keras.models import Model
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

input_layer = Input(shape=(20,))
print(input_layer)

hidden_layer1 = Dense(64, activation='relu')(input_layer) 
hidden_layer2 = Dense(64, activation='relu')(hidden_layer1) 

output_layer = Dense(1, activation='sigmoid')(hidden_layer2) 

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


X_train = np.random.rand(1000, 20) 
y_train = np.random.randint(2, size=(1000, 1)) 
model.fit(X_train, y_train, epochs=10, batch_size=32) 



X_test = np.random.rand(200, 20) 
y_test = np.random.randint(2, size=(200, 1)) 
loss, accuracy = model.evaluate(X_test, y_test) 
print(f'Test loss: {loss}') 
print(f'Test accuracy: {accuracy}') 


# Dropout layer

input_layer = Input(shape=(20,))

hidden_layer = Dense(64, activation='relu')(input_layer)

# Add a Dropout layer
dropout_layer = Dropout(rate=0.5)(hidden_layer)
hidden_layer2 = Dense(64, activation='relu')(dropout_layer)

output_layer = Dense(1, activation='sigmoid')(hidden_layer2)
model = Model(inputs=input_layer, outputs=output_layer)

model.summary()



# Batch Normalization

input_layer = Input(shape=(20,))

hidden_layer = Dense(64, activation='relu')(input_layer)

# Add a BatchNormalization layer
batch_norm_layer = BatchNormalization()(hidden_layer)
hidden_layer2 = Dense(64, activation='relu')(batch_norm_layer)
output_layer = Dense(1, activation='sigmoid')(hidden_layer2)
model = Model(inputs=input_layer, outputs=output_layer)

model.summary()




input_layer = Input(shape=(20,))

# hidden layers with dropout
hidden_layer1 = Dense(64, activation='relu')(input_layer)
dropout1 = Dropout(0.5)(hidden_layer1)
hidden_layer2 = Dense(64, activation='relu')(dropout1)
dropout2 = Dropout(0.5)(hidden_layer2)
output_layer = Dense(1, activation='sigmoid')(dropout2)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')


input_layer = Input(shape=(20,))

# Tanh activation
hidden_layer1 = Dense(64, activation='tanh')(input_layer)
hidden_layer2 = Dense(64, activation='tanh')(hidden_layer1)
output_layer = Dense(1, activation='sigmoid')(hidden_layer2)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
