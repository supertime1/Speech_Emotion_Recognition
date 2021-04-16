import numpy as np
import sklearn
import librosa


class AudioProcessor:
    def __init__(self, sample_freq, slice_span, overlap_ratio, n_mels):
        self.sample_freq = sample_freq
        self.slice_span = slice_span
        self._nperseg = int(slice_span / 1000 * sample_freq)
        self._n_fft = int(pow(2, np.ceil(np.log(self._nperseg) / np.log(2))))
        self._hop_length = int(self._n_fft * (1 - overlap_ratio))
        self.n_mels = n_mels

    def _add_nomralization(self, func):
        def wrapper(*args):
            scaled_data = sklearn.preprocessing.minmax_scale(*args, (-1, 1))
            return func(scaled_data)
        return wrapper()

    @_add_nomralization
    def spectrogram(self, data):
        spec = np.abs(librosa.stft(data, n_fft=self._n_fft,
                                   hop_length=self._hop_length))
        return spec

    @_add_nomralization
    def mel_spectrogram(self, data):
        mel_spec = librosa.feature.melspectrogram(data,
                                                  sr=self.sample_freq,
                                                  n_fft=self._n_fft,
                                                  hop_length=self._hop_length,
                                                  n_mels=self.n_mels)
        return mel_spec

