import pickle
import numpy as np
# from layers import Linear, LeakyReLU
# from nn import Model
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist as MNIST
import matplotlib.pyplot as plt 

# Load Dataset
(X_train, y_train), (X_test, y_test) = MNIST.load_data()

# normalization
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# 4D (60000, 28, 28, 1) tensor to 2D (60000, 784) tensor 
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# define Neural Network Architecture
'''def ANN(X):
    model = Model()
    model.add(Linear(X.shape[1], 128))
    model.add(LeakyReLU())
    model.add(Linear(128, 256))
    model.add(LeakyReLU())
    model.add(Linear(256, 10))
    return model

# instantiate the NN model
model = ANN(X_train)

# train
hist = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), learning_rate=0.1, verbose=1)'''

def ANN(X):
    model = Sequential()
    model.add(Dense(128, input_shape=(X.shape[1], ), activation=LeakyReLU(alpha=0.2)))
    model.add(Dense(256, activation=LeakyReLU(alpha=0.2)))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = ANN(X_train)

h = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# plot results (Accuracy & Loss)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

ax1.plot(h.history['loss'], label='Train Loss')
ax1.plot(h.history['val_loss'], label='Validation Loss')
ax1.set(xlabel='Epochs', ylabel='Loss')
ax1.set_title('Loss Metrics')
ax1.legend(loc='best')
ax1.grid()

ax2.plot(h.history['accuracy'], label='Train Acc')
ax2.plot(h.history['val_accuracy'], label='Validation Acc')
ax2.set(xlabel='Epochs', ylabel='Acc')
ax2.set_title('Acc Metrics')
ax2.legend(loc='best')
ax2.grid()

fig.tight_layout()
plt.show()

model.save('models/MNIST_tf.h5')
model.save_weights('models/MNIST_tf_weights.h5')

# # save the model
# with open('models/MNIST.dat', 'wb') as f:
#     pickle.dump(model, f)

# # simple test
# for i in range(10, 21):
#     print(f"Predicted: {model.predict(X_test[i])} | Actual: {y_test[i]}")   