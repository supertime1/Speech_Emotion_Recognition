import argparse
from data_handler import *
from audio_processor import AudioProcessor
import tensorflow as tf
from model.model_utils import *
from model.models.mobile_net import MobileNet
from model.models.simple_cnn import simple_cnn
from tensorflow.keras.layers.experimental import preprocessing
# global variable
FLAGS = None


def main():
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
    )
    data_handler = DataHandler(FLAGS.raw_data_path, FLAGS.train_ratio,
                               FLAGS.val_ratio, FLAGS.res_freq, FLAGS.block_span,
                               FLAGS.stride_span, FLAGS.random_seed, FLAGS.db_name)

    audio_processor = AudioProcessor(FLAGS.res_freq, FLAGS.slice_span,
                                     FLAGS.overlap_ratio, FLAGS.n_mels,
                                     FLAGS.snr)

    # get num of classes
    classes_lst = data_handler.count_label()
    num_classes = len(classes_lst)
    print(f'\n\nThere are {num_classes} classes in total....\n\n')

    # calculate the input shape, which is needed to initialize keras model
    sample_data = np.random.rand((FLAGS.block_span * FLAGS.res_freq))
    sample, _ = audio_processor.get_spectrogram(sample_data, 1)
    sample = np.expand_dims(sample, -1)
    input_shape = sample.shape
    print(f'\n\nInput data shape: {input_shape}...\n\n')

    # get the train and validation filenames
    train_path = os.path.join(FLAGS.db_name, 'data', 'train')
    val_path = os.path.join(FLAGS.db_name, 'data', 'val')
    train_filenames, _ = data_handler.get_filenames_tensor(train_path)
    val_filenames, _ = data_handler.get_filenames_tensor(val_path)

    # tf.data pipeline
    def preprocess_dataset(files):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(data_handler.get_waveform_and_label_tensor,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
        output_ds = output_ds.map(audio_processor.get_spectrogram_tensor,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return output_ds

    train_ds = preprocess_dataset(train_filenames)
    val_ds = preprocess_dataset(val_filenames)
    train_ds = train_ds.batch(FLAGS.batch_size)
    val_ds = val_ds.batch(FLAGS.batch_size)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

    # define callbacks
    # early stop
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=10,
                                                  restore_best_weights=True)
    # learning rate decay callback
    #lr_schedule = tf.keras.callbacks.LearningRateScheduler(decay)
    callback_list = [early_stop]

    # start training
    with strategy.scope():
        # create a normalization layer by using the training data
        norm_layer = preprocessing.Normalization()
        norm_layer.adapt(train_ds.map(lambda x, _: x))
        model = MobileNet(input_shape=input_shape, alpha=0.5,
                          norm_layer=norm_layer, classes=num_classes, dropout=0.8)
        model.summary()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        history = model.fit(train_ds,
                            epochs=FLAGS.epochs,
                            validation_data=val_ds,
                            verbose=1,
                            callbacks=callback_list
                            )

        model.save('artifacts/models/' + FLAGS.model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Flags to configure DataHandler
    parser.add_argument('--raw_data_path', action='store',
                        default='RAVDESS/raw_data',
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
    # Flags to configure training
    parser.add_argument('--model_name', action='store',
                        default='cnn',
                        help='select dnn model type', type=str)
    parser.add_argument('--batch_size', action='store',
                        default=128,
                        help='training batch size', type=int)
    parser.add_argument('--epochs', action='store',
                        default=100,
                        help='training epochs', type=int)

    FLAGS = parser.parse_args()

    main()
