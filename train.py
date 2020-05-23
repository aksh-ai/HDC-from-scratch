import pickle
import numpy as np
from layers import Linear, LeakyReLU
from nn import Model
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
def ANN(X):
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
hist = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), learning_rate=0.1, verbose=1)

# plot results (Accuracy & Loss)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

ax1.plot(hist['loss'], label='Train Loss')
ax1.plot(hist['val_loss'], label='Validation Loss')
ax1.set(xlabel='Epochs', ylabel='Loss')
ax1.set_title('Loss Metrics')
ax1.legend(loc='best')
ax1.grid()

ax2.plot(hist['accuracy'], label='Train Acc')
ax2.plot(hist['val_accuracy'], label='Validation Acc')
ax2.set(xlabel='Epochs', ylabel='Acc')
ax2.set_title('Acc Metrics')
ax2.legend(loc='best')
ax2.grid()

fig.tight_layout()
plt.show()

# save the model
with open('models/MNIST.dat', 'wb') as f:
    pickle.dump(model, f)