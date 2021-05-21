import numpy as np
import sklearn
import librosa
import tensorflow as tf
import math
import matplotlib
from matplotlib import cm


class AudioProcessor:
    def __init__(self, sample_freq, slice_span, overlap_ratio, n_mels, snr):
        self.sample_freq = sample_freq
        self.slice_span = slice_span
        self.overlap_ratio = overlap_ratio
        self.n_mels = n_mels
        self.snr = snr

    @property
    def n_per_seg(self):
        return int(self.slice_span / 1000 * self.sample_freq)

    @property
    def n_fft(self):
        return int(pow(2, np.ceil(np.log(self.n_per_seg) / np.log(2))))

    @property
    def hop_length(self):
        return int(self.n_fft * (1 - self.overlap_ratio))

    def add_additive_white_gaussian_noise(self, data, label):
        rms_signal = math.sqrt(np.mean(data ** 2))
        std_noise = abs(rms_signal) / math.sqrt(10 ** (self.snr / 10))
        noise = np.random.normal(0, std_noise, data.shape[0])
        data = data + noise
        return data, label

    def get_add_additive_white_gaussian_noise_tensor(self, data, label):
        data, label = tf.py_function(self.add_additive_white_gaussian_noise,
                                     inp=[data, label],
                                     Tout=[tf.float32, tf.float32])
        data.set_shape(data.shape)
        return data, label

    def get_spectrogram(self, data, label):
        spec = np.abs(librosa.stft(np.asarray(data), n_fft=self.n_fft,
                                   hop_length=self.hop_length))
        return spec, label

    def get_spectrogram_tensor(self, data, label):
        spectrogram, label = tf.py_function(self.get_spectrogram, inp=[data, label],
                                            Tout=[tf.float32, tf.float32])
        spectrogram = tf.expand_dims(spectrogram, -1)

        spectrogram.set_shape(spectrogram.shape)
        return spectrogram, label

    def get_mel_spectrogram(self, data, label):
        mel_spec = librosa.feature.melspectrogram(np.asarray(data),
                                                  sr=self.sample_freq,
                                                  n_fft=self.n_fft,
                                                  hop_length=self.hop_length,
                                                  n_mels=self.n_mels)
        return mel_spec, label

    def get_mel_spectrogram_tensor(self, data, label):
        mel_spec, label = tf.py_function(self.get_mel_spectrogram, inp=[data, label],
                                         Tout=[tf.float32, tf.float32])
        mel_spec = tf.expand_dims(mel_spec, -1)

        mel_spec.set_shape(mel_spec.shape)
        return mel_spec, label

    def get_mfcc(self, data, label):
        mel_spec, label = self.get_mel_spectrogram(data, label)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec))
        return mfcc, label

    def get_mfcc_tensor(self, data, label):
        mfcc, label = tf.py_function(self.get_mfcc, inp=[data, label],
                                     Tout=[tf.float32, tf.float32])
        return mfcc, label

    @staticmethod
    def spectrogram_to_rgb(spectrogram, label):
        """
        Convert spectrogram/mel spectrogram (in amplitude) to rgb image (0-255)
        :param spectrogram: spectrogram/mel spectrogram
        :param label: emotion label of the spectrogram
        :return: rgb image and its corresponding emotion label
        """
        sm = cm.ScalarMappable(norm=None, cmap='jet')
        color_spectrogram = sm.to_rgba(spectrogram, norm=False, bytes=True)
        return color_spectrogram[:, :, :3], label

    def spectrogram_to_rgb_tensor(self, spectrogram, label):
        """
        TF-graph version of spectrogram_to_rgb
        """
        spectrogram = np.squeeze(spectrogram, -1)
        color_spectrogram, label = tf.py_function(self.spectrogram_to_rgb,
                                                  inp=[spectrogram, label],
                                                  Tout=[tf.float32, tf.float32])
        return color_spectrogram, label

    # TODO: Speech features
    # 1. Prosody features: fundamental frequency:F0, speaking rate
    # 2. Spectral features: linear prediction cepstral coefficients (LPCC)
    # 3. Voice quality features: jitter, shimmer, normalized amplitude quotient (NAQ)
    # 4. Others: Energy?
