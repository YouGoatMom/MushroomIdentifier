import tensorflow as tf
import keras

class Trainer():
    def train(self, data, model, n_epochs):
        model.fit(data.dataset, epochs = n_epochs, batch_size= 32)