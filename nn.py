import time
import numpy as np
from losses import *

class Model:
    def __init__(self, layers=[]):
        self.layers = layers

        self.__loss = categorical_crossentropy
        self.__grad_loss = grad_categorical_crossentropy

    def add(self, layer):
        self.layers.append(layer)

    def __forward(self, inputs):
        try:
            acts = []
            inps = inputs

            for layer in self.layers:
                acts.append(layer(inps))
                inps = acts[-1]

            return acts

        except Exception as e:
            print(f'Error forwarding through layers\n{e}')        

    def __train(self, X, y, learning_rate):
        layer_acts = self.__forward(X)
        layer_inps = [X] + layer_acts

        outputs = layer_acts[-1]

        loss = self.__loss(outputs, y)
        grad_loss = self.__grad_loss(outputs, y)

        for idx in range(len(self.layers))[::-1]:
            layer = self.layers[idx]
            grad_loss = layer.backward(layer_inps[idx], grad_loss, learning_rate)

        return np.mean(loss)    

    def fit(self, X_train, y_train, epochs=30, batch_size=32, learning_rate=0.1, validation_data=(), verbose=1):
        print(f"{int(len(X_train)/batch_size)} batch of samples per epoch | Total of {len(X_train)} samples per epoch")

        train_acc, val_acc = [], []
        train_loss, val_loss = [], []

        t_loss = None
        t_acc = None

        X_test, y_test = validation_data[0], validation_data[1]

        num_samples = len(X_train)

        start_time= time.time()

        for e in range(epochs):
            e_start = time.time()

            for offset in range(0, num_samples, batch_size):
                end = offset + batch_size
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]

                t_loss = self.__train(batch_x, batch_y, learning_rate)

            train_acc.append(np.mean(self.predict(X_train) == y_train))    
            val_acc.append(np.mean(self.predict(X_test) == y_test))

            train_loss.append(t_loss)
            val_loss.append(self.validate(X_test, y_test))

            e_end = (time.time() - e_start) / 60

            if (e == 0) or (e == (epochs-1)) or (e % verbose == 0):
                print(f"Epoch {e+1}")
                print(f"accuracy: {train_acc[-1]:.4f} - loss: {train_loss[-1]:.4f} - val_accuracy: {val_acc[-1]:.4f} - val_loss: {val_loss[-1]:.4f} - duration: {e_end:.2f} mins")

        print(f"Total Training Duration: {(time.time() - start_time)/60:.2f} mins")

        return {"accuracy": train_acc, "val_accuracy": val_acc, "loss": train_loss, "val_loss": val_loss}

    def predict(self, X):
        predictions = self.__forward(X)[-1]
        return predictions.argmax(axis=-1)

    def validate(self, X, y):
        predictions = self.__forward(X)[-1]  
        loss = self.__loss(predictions, y)
        return np.mean(loss)