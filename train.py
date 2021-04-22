import argparse
from data_handler import *
from audio_processor import AudioProcessor
import tensorflow as tf
from model.model_util import *

# global variable
FLAGS = None


def main():
    data_handler = DataHandler(FLAGS.raw_data_path, FLAGS.train_ratio,
                               FLAGS.val_ratio, FLAGS.res_freq, FLAGS.block_span,
                               FLAGS.stride_span, FLAGS.random_seed)

    audio_processor = AudioProcessor(FLAGS.res_freq, FLAGS.slice_span,
                                     FLAGS.overlap_ratio, FLAGS.n_mels,
                                     FLAGS.snr)

    sample_data = np.random.rand((FLAGS.block_span * FLAGS.res_freq))
    sample_mel, _ = audio_processor.mel_spectrogram(sample_data, 1)
    sample_mel = np.expand_dims(sample_mel, -1)
    input_shape = sample_mel.shape
    print(input_shape)

    train_filenames, train_num_samples = data_handler.get_filenames('data/train')
    val_filenames, val_num_samples = data_handler.get_filenames('data/val')

    def preprocess_dataset(files):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(data_handler.get_waveform_and_label,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
        output_ds = output_ds.map(audio_processor.get_mel_tensor,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return output_ds

    train_ds = preprocess_dataset(train_filenames)
    val_ds = preprocess_dataset(val_filenames)
    train_ds = train_ds.batch(FLAGS.batch_size)
    val_ds = val_ds.batch(FLAGS.batch_size)
    train_ds = train_ds.cache().prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.experimental.AUTOTUNE)

    # early stop
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=20,
                                                  restore_best_weights=True)
    # learning rate decay callback
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(decay)
    callback_list = [early_stop, lr_schedule]

    model = MobileNet(input_shape=input_shape, classes=8)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    history = model.fit(train_ds,
                        epochs=FLAGS.epochs,
                        validation_data=val_ds,
                        verbose=1,
                        callbacks=callback_list
                        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', action='store',
                        default='raw_data',
                        help='raw data file path', type=str)
    parser.add_argument('--train_ratio', action='store',
                        default=0.9,
                        help='train data ratio in all data files', type=float)
    parser.add_argument('--val_ratio', action='store',
                        default=0.2,
                        help='validation data ratio in train data files', type=float)
    parser.add_argument('--res_freq', action='store',
                        default=16000,
                        help='re-sampling frequency (Hz)', type=int)
    parser.add_argument('--block_span', action='store',
                        default=1,
                        help='block time span in second (e.g. 1s)', type=float)
    parser.add_argument('--stride_span', action='store',
                        default=30,
                        help='stride time span in millisecond (e.g. 30ms)', type=int)
    parser.add_argument('--random_seed', action='store',
                        default=10,
                        help='random seed in splitting data into train and test', type=int)
    # continue here
    parser.add_argument('--slice_span', action='store',
                        default=16,
                        help='stft window in millisecond', type=int)
    parser.add_argument('--overlap_ratio', action='store',
                        default=0.75,
                        help='stft window overlap ratio', type=float)
    parser.add_argument('--val_ratio', action='store',
                        default=0.1,
                        help='validation data ratio in train data files', type=float)
    parser.add_argument('--res_freq', action='store',
                        default=16000,
                        help='re-sampling frequency (Hz)', type=int)
    parser.add_argument('--n_mels', action='store',
                        default=64,
                        help='number mel filterbanks', type=int)
    parser.add_argument('--snr', action='store',
                        default=20,
                        help='signal-to-noise in dB if noise is added', type=int)
    parser.add_argument('--batch_size', action='store',
                        default=64,
                        help='training batch size', type=int)
    parser.add_argument('--epochs', action='store',
                        default=100,
                        help='training epochs', type=int)

    FLAGS = parser.parse_args()

    main()
