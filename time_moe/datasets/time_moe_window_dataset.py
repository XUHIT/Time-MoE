#!/usr/bin/env python
# -*- coding:utf-8 _*-
import random
import numpy as np

from time_moe.datasets.ts_dataset import TimeSeriesDataset


class TimeMoEWindowDataset:
    """
TimeMoEWindowDataset 是一个用于从时间序列数据中构造滑动窗口样本的数据集类。

该类的主要用途是将原始的时间序列数据切分为长度固定的子序列（滑动窗口），
以满足诸如 Transformer 或 LSTM 等深度学习模型的输入格式要求。
特别适用于输入和标签都为定长的监督学习任务。

该类默认使用**非重叠滑窗**，即窗口之间的步长 stride 与窗口长度相等；
但也支持用户自定义滑动步长（stride < context_length），以获得更多样本。

【主要属性说明】：
    - dataset (TimeSeriesDataset)：原始时间序列数据集。
    - context_length (int)：模型每次看到的历史长度（滑窗输入长度）。
    - prediction_length (int)：模型每次需要预测的未来长度，默认为 0（即无标签监督）。
    - window_size (int)：滑动窗口的总长度，等于 context_length + prediction_length。
    - window_size_plus_one (int)：滑动窗口长度加一，用于某些对齐计算。
    - stride (int)：滑窗的步长，控制滑动的粒度。若不指定则默认为 window_size，即不重叠滑窗。
    - sub_seq_indexes (List[Tuple[int, int]])：保存了每个生成样本在原数据中所属序列及其偏移的索引元组。

【主要方法说明】：
    - __len__()：返回所有滑窗样本的总数。
    - __getitem__(idx)：根据下标返回一个滑窗样本，包括输入 input_ids、标签 labels、损失掩码 loss_masks。
    - __iter__()：支持迭代访问所有样本。

【使用示例】：
    >>> dataset = TimeSeriesDataset(...)  # 已准备好的时间序列数据集
    >>> context_length = 96
    >>> prediction_length = 16
    >>> window_dataset = TimeMoEWindowDataset(dataset, context_length, prediction_length, stride=32)
    >>> for sample in window_dataset:
    >>>     print(sample["input_ids"], sample["labels"], sample["loss_masks"])

通过调整 context_length、prediction_length 和 stride 参数，
用户可以自由控制滑窗的覆盖范围和样本数量，以适应不同的任务需求。
"""
   
    def __init__(self, dataset: TimeSeriesDataset, context_length: int, prediction_length: int = 0, stride: int = None, **kwrags):

        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length
        self.window_size_plus_one = self.window_size + 1
        self.stride = stride if stride else self.window_size

        num_seqs = len(self.dataset)
        iterator = range(num_seqs)
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=num_seqs)
        except ImportError:
            pass
        self.sub_seq_indexes = []
        for seq_idx in iterator:
            n_points = self.dataset.get_sequence_length_by_idx(seq_idx)
            # Skip sequences with fewer than 2 points
            if n_points < 2:
                continue
            # Calculate the number of windows for this sequence
            self.sub_seq_indexes.append((seq_idx, 0))
            for offset_idx in range(
                self.stride,
                n_points - self.window_size_plus_one + 1,
                self.stride
            ):
                self.sub_seq_indexes.append((seq_idx, offset_idx))

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, seq_idx):
        seq_i, offset_i = self.sub_seq_indexes[seq_idx]
        seq = self.dataset[seq_i][offset_i: offset_i + self.window_size_plus_one]
        seq = np.array(seq, dtype=np.float32)

        loss_mask = np.ones(len(seq) - 1, dtype=np.int32)
        n_pad = self.window_size_plus_one - len(seq)
        if n_pad > 0:
            seq = np.pad(seq, (0, n_pad), 'constant', constant_values=0)
            loss_mask = np.pad(loss_mask, (0, n_pad), 'constant', constant_values=0)

        return {
            'input_ids': seq[:-1],
            'labels': seq[1:],
            'loss_masks': loss_mask
        }


class UniversalTimeMoEWindowDataset:
    """
    A dataset that generates windows of time series data with pack technique.
    """
    def __init__(self, dataset: TimeSeriesDataset, context_length: int, prediction_length: int = 0,
                 shuffle: bool = False, **kwrags):
        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length

        self.window_info_list = []
        n_seqs = len(self.dataset)

        cur_window_info = []
        num_cur_remaining_points = self.window_size

        iterator = range(n_seqs)
        if shuffle:
            iterator = list(iterator)
            random.shuffle(iterator)

        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=n_seqs)
        except ImportError:
            pass

        for seq_idx in iterator:
            seq_len = self.dataset.get_sequence_length_by_idx(seq_idx)
            remaining_seq_len = seq_len
            while remaining_seq_len > 0:
                if remaining_seq_len < num_cur_remaining_points:
                    cur_window_info.append(
                        (seq_idx, seq_len - remaining_seq_len, remaining_seq_len)
                    )

                    # update states
                    num_cur_remaining_points -= remaining_seq_len
                    remaining_seq_len = 0
                else:
                    # add the part of this seq to cur_window
                    cur_window_info.append(
                        (seq_idx, seq_len - remaining_seq_len, num_cur_remaining_points)
                    )

                    # update states
                    remaining_seq_len -= num_cur_remaining_points
                    self.window_info_list.append(cur_window_info)

                    # reset current window
                    num_cur_remaining_points = self.window_size
                    cur_window_info = []

        if num_cur_remaining_points > 0:
            # drop last batch for speed-up
            pass

    def __len__(self):
        return len(self.window_info_list)

    def __getitem__(self, window_idx):
        window_info = self.window_info_list[window_idx]
        seq = []
        for seq_idx, start_idx_in_seq, offset in window_info:
            part_seq = self.dataset[seq_idx][start_idx_in_seq: start_idx_in_seq + offset]
            seq.append(part_seq)
        if len(seq) == 1:
            seq = seq[0]
            if not isinstance(seq, np.ndarray):
                seq = np.array(seq, dtype=np.float32)
            else:
                seq = seq.astype(np.float32)
        else:
            seq = np.concatenate(seq, axis=0, dtype=np.float32)
        return {
            'input_ids': seq[:-1],
            'labels': seq[1:],
        }
