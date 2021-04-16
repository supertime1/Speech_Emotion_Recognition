import os
import pickle
import numpy as np
import wavio
import argparse


def prepare_dataset(folder_path, save_path, block_span, stride_span, channel_idx):
    for folder in os.listdir(folder_path):
        # e.g. folder: Actor_01
        print(f'Processing {folder}')
        data_path = os.path.join(folder_path, folder)
        fn_lst = os.listdir(data_path)
        for idx in range(len(fn_lst)):
            # e.g. file: 03-01-08-02-01-02-01.wav
            file_data_lst = []
            file_label_lst = []
            wave = wavio.read(os.path.join(data_path, fn_lst[idx]))
            signal = wave.data[:, channel_idx]

            # convert the time span (s or ms) to length (number of samples)
            block_len = wave.rate * block_span
            stride_len = int(wave.rate * stride_span / 1000)

            for i in range(0, len(signal), stride_len):
                if i + block_len > len(signal):
                    break
                block_signal = signal[i:i + block_len]
                # extract both emotion and emotion intensity as labels
                block_label = [int(fn_lst[idx][6:8]), int(fn_lst[idx][9:11])]
                file_data_lst.append(block_signal)
                file_label_lst.append(block_label)

            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if not os.path.exists(save_path + '/' + folder):
                os.mkdir(save_path + '/' + folder)

            pickle.dump(np.asarray(file_data_lst),
                        open(save_path + '/' + folder + '/data_' + str(idx), 'wb'))
            pickle.dump(np.asarray(file_label_lst),
                        open(save_path + '/' + folder + '/label_' + str(idx), 'wb'))


if __name__ == '__main__':

    folder_path = 'raw_data'
    save_path = 'data'

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

    prepare_dataset(folder_path, save_path, block_span, stride_span, channel_idx)





