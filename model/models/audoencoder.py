from tensorflow.keras.layers import  Input, Conv1D, Conv1DTranspose
import tensorflow as tf
from tensorflow.keras.models import Model


class Autoencoder(Model):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            Input(shape=input_shape),
            Conv1D(128, 3, activation='relu', padding='same', strides=1),
            Conv1D(64, 3, activation='relu', padding='same', strides=1),
            Conv1D(32, 3, activation='relu', padding='same', strides=1),
        ])

        self.decoder = tf.keras.Sequential([
            Conv1DTranspose(32, kernel_size=3, strides=1, activation='relu', padding='same'),
            Conv1DTranspose(64, kernel_size=3, strides=1, activation='relu', padding='same'),
            Conv1DTranspose(128, kernel_size=3, strides=1, activation='relu', padding='same'),
            Conv1D(1, kernel_size=3, activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
