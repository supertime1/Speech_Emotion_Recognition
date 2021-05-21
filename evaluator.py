from evaluation_tools.confusion_matrix import *
import sklearn
from data_handler import DataHandler
from audio_processor import AudioProcessor
import argparse
import os
import tensorflow as tf
from tensorflow.keras.models import load_model


class Evaluator:
    def __init__(self, model: tf.keras.Model, db_name: str, ds_name: str, audio_processor: AudioProcessor):
        self.model = model
        self.db_name = db_name
        self.ds_name = ds_name
        self.audio_processor = audio_processor

    def evaluate_on_test(self):
        test_path = os.path.join(self.db_name, 'data', self.ds_name)
        test_fn, _ = DataHandler.get_filenames(test_path)
        test_ds = [DataHandler.get_waveform_and_label(_) for _ in test_fn]
        test_ds = [self.audio_processor.get_spectrogram(waveform, label) for
                                           waveform, label in test_ds]
        test_data_lst, test_label_lst = zip(*[audio_processor.spectrogram_to_rgb(spec, label) for
                                              spec, label in test_ds])

        test_data = np.asarray(test_data_lst)
        test_label = np.asarray(test_label_lst)
        test_pred_raw = self.model.predict(test_data)
        test_pred = np.argmax(test_pred_raw, axis=-1)
        cm = sklearn.metrics.confusion_matrix(test_label, test_pred)
        # W: anger; L: boredom; E: disgust; A: anxiety/fear; F: happiness; T: sadness;
        # N: neutral
        class_names = [None]
        if self.db_name == 'EMODB':
            class_names = ['Anger', 'Boredom', 'Disgust', 'Anxiety',
                           'Happiness', 'Sadness', 'Neutral']

        figure = plot_confusion_matrix(cm, class_names=class_names, normalize=True)
        figure.savefig('artifacts/figures/confusion_matrix.png')

        f1_score = sklearn.metrics.f1_score(test_label, test_pred, average='weighted')
        print(f'Weighted F1 score is {f1_score}')


# TODO: add miss prediction data analysis (e.g. noise, gender difference, age, volumen difference, devices, accents),
#  what fraction of the errors has for say female; of all data with female, what fraction is misclassified;
#  what fraction of all the data is from female; how much room for improvement is there on data from female;


# TODO: Compare with human level performance (HLP), analyze how much room to improve; HLP can be established
#  by testing healthy human's ability to identify the emotions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Flags to configure DataHandler
    parser.add_argument('--db_name', action='store',
                        default='EMODB',
                        help='Database name to be processed', type=str)
    # Flags to configure AudioProcessor
    parser.add_argument('--sample_freq', action='store',
                        default=16000,
                        help='sampling frequency (Hz)', type=int)
    parser.add_argument('--slice_span', action='store',
                        default=16,
                        help='stft window in millisecond', type=int)
    parser.add_argument('--overlap_ratio', action='store',
                        default=0.75,
                        help='stft window overlap ratio', type=float)
    parser.add_argument('--n_mels', action='store',
                        default=64,
                        help='number mel filterbanks', type=int)
    parser.add_argument('--snr', action='store',
                        default=20,
                        help='signal-to-noise in dB if noise is added', type=int)

    parser.add_argument('--model_dir', action='store',
                        default='artifacts/models/ft_mobile_net_v2',
                        help='select which modelzoo to evaluate', type=str)
    # Flags to configure Evaluator
    parser.add_argument('--ds_name', action='store',
                        default='test',
                        help='Dataset name to be evaluated', type=str)
    FLAGS = parser.parse_args()

    model = load_model(FLAGS.model_dir)
    audio_processor = AudioProcessor(FLAGS.sample_freq, FLAGS.slice_span, FLAGS.overlap_ratio,
                                     FLAGS.n_mels, FLAGS.snr)
    evaluator = Evaluator(model, FLAGS.db_name, FLAGS.ds_name, audio_processor)
    evaluator.evaluate_on_test()
