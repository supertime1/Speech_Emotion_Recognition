import os
import numpy as np
import librosa
import tensorflow as tf
import argparse
import soundfile as sf


class DataHandler:

    def __init__(self, raw_data_path: str, train_ratio: float, val_ratio: float, res_freq: int,
                 block_span: int, stride_span: int, random_seed: int, db_name: str):
        self.raw_data_path = raw_data_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.res_freq = res_freq
        self.block_span = block_span
        self.stride_span = stride_span
        self.random_seed = random_seed
        self.fn_dic = {}
        self.train_fn_dic = {}
        self.val_fn_dic = {}
        self.test_fn_dic = {}
        self.db_name = db_name
        self._create_block_fn()

    def _create_block_fn(self):
        """
        :return: generators and dictionaries containing train, validation and test data
        """
        if self.db_name == 'RAVDESS':
            # e.g. folder: Actor_01
            for folder in os.listdir(self.raw_data_path):
                # e.g. data_path: data/Actor_01
                data_path = os.path.join(self.raw_data_path, folder)
                fn_lst = os.listdir(data_path)
                for idx in range(len(fn_lst)):
                    # e.g. fn_path: data/Actor_01/03-01-01-01-01-01-02.wav
                    fn_path = os.path.join(data_path, fn_lst[idx])
                    label = int(fn_lst[idx][6:8]) - 1
                    if label not in self.fn_dic:
                        self.fn_dic[label] = [fn_path]
                    else:
                        self.fn_dic[label].append(fn_path)
                # fn_dic is created to be {'00':[*.wav], '01':[*.wav], ...}

                for label, fn_lst in self.fn_dic.items():
                    np.random.seed(seed=self.random_seed)
                    np.random.shuffle(fn_lst)
                    self.val_fn_dic[label] = fn_lst[:int(self.train_ratio * self.val_ratio * len(fn_lst))]
                    self.train_fn_dic[label] = fn_lst[int(self.train_ratio * self.val_ratio * len(fn_lst)):
                                                      int(self.train_ratio * len(fn_lst))]
                    self.test_fn_dic[label] = fn_lst[int(self.train_ratio * len(fn_lst)):]

        if self.db_name == 'EMODB':
            pass

    def create_label_folder(self):
        self._convert_to_block(self.train_fn_dic, 'train')
        self._convert_to_block(self.val_fn_dic, 'val')
        self._convert_to_block(self.test_fn_dic, 'test')

        # if do_train_val_split:
        #   val_path = os.path.join('data', 'val')
        #   if not os.path.exists(val_path):
        #       os.mkdir(val_path)
        #   train_path = os.path.join('data', 'train')
        #   for label_path in os.listdir(train_path):
        #       # e.g. train_label_path: data/train/0/
        #       train_label_path = os.path.join(train_path, label_path, '*.wav')
        #       fn_lst = glob.glob(train_label_path)

    #
    #       np.random.seed(seed=self.random_seed)
    #       np.random.shuffle(fn_lst)
    #       val_fn_lst = fn_lst[:int(len(fn_lst) * self.val_ratio)]
    #       target_val_fn_path = os.path.join(val_path, label_path)
    #       if not os.path.exists(target_val_fn_path):
    #           os.mkdir(target_val_fn_path)
    #       for val_fn in val_fn_lst:
    #           parts = val_fn.split(os.path.sep)
    #           wav_fn = parts[-1]
    #           shutil.move(val_fn, os.path.join(target_val_fn_path, wav_fn))

    def _convert_to_block(self, name_fn_dic, name):
        if not os.path.exists('data'):
            os.mkdir('data')

        print(f'Creating {name} data')
        # e.g. name_path: data/train
        name_path = os.path.join('data', name)
        if not os.path.exists(name_path):
            os.mkdir(name_path)

        # e.g. fn_lst: [raw_data/Actor_01/03-01-01-01-01-01-02.wav, data/Actor_02/*.wav, ...]
        for label, fn_lst in name_fn_dic.items():
            print(f'Processing label {label}')
            # e.g. label_folder_path: data/train/0
            label_folder_path = os.path.join(name_path, str(label))
            if not os.path.exists(label_folder_path):
                os.mkdir(label_folder_path)
            for i in range(len(fn_lst)):
                if i % 100 == 0:
                    print(f'working on {i}th file')
                y, sr = librosa.load(fn_lst[i], sr=self.res_freq)
                signal, _ = librosa.effects.trim(y)
                block_len = self.res_freq * self.block_span
                stride_len = int(self.res_freq * self.stride_span / 1000)
                for j in range(0, len(signal), stride_len):
                    if j + block_len > len(signal):
                        break
                    block_signal = signal[i:i + block_len]
                    sf.write(label_folder_path + '/' + fn_lst[i][18:-4] + '_' + str(j) + '.wav',
                             block_signal, self.res_freq)

    def get_waveform_and_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = tf.strings.to_number(parts[-2], out_type=tf.int32)
        label = tf.one_hot(label, 8)
        audio_binary = tf.io.read_file(file_path)
        waveform, _ = tf.audio.decode_wav(audio_binary)
        waveform = tf.reshape(waveform, [self.block_span * self.res_freq])
        return waveform, label

    @staticmethod
    def get_filenames(data_dir):
        filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
        filenames = tf.random.shuffle(filenames)
        num_samples = len(filenames)
        print('Number of total examples:', num_samples)
        print('Example file tensor:', filenames[0])
        return filenames, num_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', action='store',
                        default='raw_data',
                        help='raw data file path', type=str)
    parser.add_argument('--tr', action='store',
                        default=0.9,
                        help='train data ratio in all data files', type=float)
    parser.add_argument('--vr', action='store',
                        default=0.1,
                        help='validation data ratio in train data files', type=float)
    parser.add_argument('--rs', action='store',
                        default=16000,
                        help='re-sampling frequency (Hz)', type=int)
    parser.add_argument('--bs', action='store',
                        default=1,
                        help='block time span in second (e.g. 1s)', type=float)
    parser.add_argument('--ss', action='store',
                        default=10,
                        help='stride time span in millisecond (e.g. 30ms)', type=int)
    parser.add_argument('--sd', action='store',
                        default=10,
                        help='random seed in splitting data into train and test', type=int)
    parser.add_argument('--db', action='store',
                        default='RAVDESS',
                        help='Database name to be processed', type=str)
    args = parser.parse_args()

    raw_data_path = args.dir
    train_ratio = args.tr
    val_ratio = args.vr
    res_freq = args.rs
    block_span = args.bs
    stride_span = args.ss
    random_seed = args.sd
    db_name = args.db_name

    data_handler = DataHandler(raw_data_path, train_ratio, val_ratio,
                               res_freq, block_span, stride_span, random_seed, db_name)
    data_handler.create_label_folder()
