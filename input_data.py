import os
import wavio


def input_data_generator(data_dir, channel_idx, block_span, stride_span):
    fn_lst = os.listdir(data_dir)
    for idx in range(len(fn_lst)):
        wave = wavio.read(os.path.join(data_dir, fn_lst[idx]))
        signal = wave.data[:, channel_idx]
        block_len = wave.rate * block_span
        stride_len = int(wave.rate * stride_span / 1000)

        for i in range(0, len(signal), stride_len):
            if i + block_len > len(signal):
                break
            block_signal = signal[i:i + block_len]
            # extract both emotion and emotion intensity as labels
            block_label = [int(fn_lst[idx][6:8]), int(fn_lst[idx][9:11])]

            yield block_signal, block_label
