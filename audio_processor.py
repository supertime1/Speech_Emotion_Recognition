import numpy as np
import sklearn
import librosa
import tensorflow as tf


class AudioProcessor:
    def __init__(self, sample_freq, slice_span, overlap_ratio, n_mels):
        self.sample_freq = sample_freq
        self.slice_span = slice_span
        self.overlap_ratio = overlap_ratio
        self.n_mels = n_mels

    @property
    def n_per_seg(self):
        return int(self.slice_span / 1000 * self.sample_freq)

    @property
    def n_fft(self):
        return int(pow(2, np.ceil(np.log(self.n_per_seg) / np.log(2))))

    @property
    def hop_length(self):
        return int(self.n_fft * (1 - self.overlap_ratio))

    def spectrogram(self, data, label):
        scaled_data = sklearn.preprocessing.minmax_scale(data, (-1, 1))
        spec = np.abs(librosa.stft(scaled_data, n_fft=self.n_fft,
                                   hop_length=self.hop_length))
        spec = sklearn.preprocessing.minmax_scale(spec, axis=1)

        return spec, label

    def mel_spectrogram(self, data, label):
        scaled_data = sklearn.preprocessing.minmax_scale(data, (-1, 1))
        mel_spec = librosa.feature.melspectrogram(scaled_data,
                                                  sr=self.sample_freq,
                                                  n_fft=self.n_fft,
                                                  hop_length=self.hop_length,
                                                  n_mels=self.n_mels)
        mel_spec = sklearn.preprocessing.minmax_scale(mel_spec, axis=1)
        return mel_spec, label

    def get_mel_tensor(self, data, label):
        mel_spec, label = tf.py_function(self.mel_spectrogram, inp=[data, label],
                                         Tout=[tf.float32, tf.float32])
        mel_spec = tf.expand_dims(mel_spec, -1)

        mel_spec.set_shape(mel_spec.shape)
        return mel_spec, label

    def get_spectrogram_tensor(self, data, label):
        spectrogram, label = tf.py_function(self.spectrogram, inp=[data, label],
                                            Tout=[tf.float32, tf.float32])
        spectrogram = tf.expand_dims(spectrogram, -1)

        spectrogram.set_shape(spectrogram.shape)
        return spectrogram, label
