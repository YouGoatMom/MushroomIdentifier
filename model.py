import tensorflow as tf
import keras

class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(256, activation='relu')
        self.dense2 = keras.layers.Dense(215)
        self.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
