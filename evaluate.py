from evaluation_tools.confusion_matrix import *
import sklearn
from data_handler import DataHandler
from audio_processor import AudioProcessor
import argparse
import os
import tensorflow as tf


def evaluate():
    audio_processor = AudioProcessor(FLAGS.res_freq, FLAGS.slice_span,
                                     FLAGS.overlap_ratio, FLAGS.n_mels,
                                     FLAGS.snr)
    # get the test data filenames
    test_path = os.path.join(FLAGS.db_name, 'data', 'test')
    test_filenames, _ = DataHandler.get_filenames(test_path)
    test_ds = [DataHandler.get_waveform_and_label(i) for i in test_filenames]
    test_data_label_lst = [audio_processor.mel_spectrogram(waveform, label)
                           for waveform, label in test_ds]
    test_data = np.asarray(test_data_label_lst)[:, 0]
    test_label = np.asarray(test_data_label_lst)[:, 1]
    # load model
    model = tf.keras.models.load_model(FLAGS.model_name)
    test_pred_raw = model.predict(test_data)
    test_pred = np.argmax(test_pred_raw, axis=-1)
    cm = sklearn.metrics.confusion_matrix(test_label, test_pred)
    # W: anger; L: boredom; E: disgust; A: anxiety/fear; F: happiness; T: sadness;
    # N: neutral
    class_names = [None]
    if FLAGS.db_name == 'EMODB':
        class_names = ['Anger', 'Boredom', 'Disgust', 'Anxiety',
                       'Happiness', 'Sadness', 'Neutral']

    figure = plot_confusion_matrix(cm, class_names=class_names, normalize=True)
    figure.savefig('artifacts/figures/confusion_matrix.png')

    f1_score = sklearn.metrics.f1_score(test_label, test_pred, average='weighted')
    print(f'Weighted F1 score is {f1_score}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Flags to configure DataHandler
    parser.add_argument('--db_name', action='store',
                        default='RAVDESS',
                        help='Database name to be processed', type=str)
    # Flags to configure AudioProcessor
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

    parser.add_argument('--model_name', action='store',
                        default='mobilenet',
                        help='select which model to evaluate', type=str)
    FLAGS = parser.parse_args()

    evaluate()
