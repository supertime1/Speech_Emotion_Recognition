import argparse
from data_handler import input_data_generator
import tensorflow as tf
from audio_processor import AudioProcessor
from model.resnet import resnet
from model.model_utils import decay

# global variable
FLAGS = None


def main():
    # 0. generator to load data
    train_data_generator = input_data_generator(FLAGS.data_dir, FLAGS.channel_idx, FLAGS.block_span, FLAGS.stride_span)
    audio_processor = AudioProcessor(FLAGS.sample_freq, FLAGS.slice_span, FLAGS.overlap_ratio)

    data_from_generator = tf.data.Dataset.from_generator(train_data_generator,
                                                         (tf.float32, tf.float32),
                                                         output_shapes=([None, 48000], [])
                                                         )
    train_data = data_from_generator.map(lambda data, label: data)
    train_label = data_from_generator.map(lambda data, label: label)
    train_data = train_data.map(audio_processor.mel_spectrogram)
    train = tf.data.Dataset.zip((train_data, train_label))
    train = train.cache()
    train = train.shuffle(2048).repeat().batch(FLAGS.batch_size, drop_remainder=True)
    train = train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ## early stop
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=20,
                                                  restore_best_weights=True)
    ## learning rate decay callback
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(decay)
    callback_list = [early_stop, lr_schedule]

    model = resnet()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.fit(train,
              epochs=FLAGS.epochs,
              steps_per_epoch=FLAGS.steps,
              verbose=1,
              callbacks=callback_list
              )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir',
        type=str,
        # pylint: disable=line-too-long
        default='data',
        # pylint: enable=line-too-long
        help='Location of speech training data archive on the web.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', action='store',
                        default=1,
                        help='block time span in second (e.g. 1s)', type=int)
    parser.add_argument('--ss', action='store',
                        default=10,
                        help='stride time span in millisecond (e.g. 10ms)', type=int)
    parser.add_argument('--ch', action='store',
                        default=0,
                        help='select channel index to process', type=int)
    args = parser.parse_args()

    block_span = args.bs
    stride_span = args.ss
    channel_idx = args.ch

    FLAGS, unparsed = parser.parse_known_args()

    # variables in input_data_generator
    FLAGS.channel_idx = None
    FLAGS.data_dir = None
    FLAGS.block_span = None
    FLAGS.stride_len = None

    # variables in class AudioProcessor
    FLAGS.sample_freq = None
    FLAGS.slice_span = None
    FLAGS.overlap_ratio = None

    # variables in keras model training
    FLAGS.batch_size = None
    FLAGS.epochs = None
    FLAGS.steps = None
