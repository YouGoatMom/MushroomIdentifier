import tensorflow as tf

class MushroomDataset():
    def __init__(self, dir):
        self.dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dir,
        labels='inferred',
        batch_size = 32
    )