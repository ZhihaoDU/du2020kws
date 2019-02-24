# coding = utf-8
import numpy as np
from scipy.io import wavfile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import ctypes
import librosa

import scipy.io as sio

class FrameDataset(Dataset):

    @staticmethod
    def read_path_list(list_file_path):
        f = open(list_file_path, 'r')
        file_list = f.readlines()
        for i in range(len(file_list)):
            file_list[i] = file_list[i].replace('\n', '')
        return np.array(file_list)

    @staticmethod
    def get_id(utt_path):
        ll = utt_path.split('/')
        ll = ll[-1].split('.')
        ll = ll[0].split('_')
        return ll[0] + '_' + ll[1]

    @staticmethod
    def get_label(utt_path):
        ll = utt_path.split('/')
        return ll[-1]

    def __init__(self, mix_dbs, noise_scp, clean_scp, is_train, rs_method) -> None:
        '''
        这个类得到的是纯净语音和带噪语音分帧后的结果。这个类只能在CPU上工作。

        mix_dbs:    需要合成哪些信噪比，一个list，比如[-6,-3,0,3,6]
        noise_scp:  噪声列表的文件，每行一个噪声（绝对路径）
        clean_scp:  纯净语音列表的文件，每行一句纯净语音（绝对路径）
        is_train:   是否在训练阶段，如果True，则用噪声的前半段随机截取一段进行混合，如果False，则用噪声的后半段
        rs_method:  使用什么rescale method，可以设置为None
        使用示例：
        #训练集：
        train_dataset = FrameDataset([-6,-3,0,3,6], 'path/to/your/noise.lst', 'path/to/your/tr_clean_speech.lst', True, None)
        #验证集：
        valid_dataset = FrameDataset([-6,-3,0,3,6,9], 'path/to/your/noise.lst', 'path/to/your/dt_clean_speech.lst', False, None)
        '''
        super().__init__()
        self.sample_freq = 16000                        # wav文件的采样率
        self.win_len = 30 * self.sample_freq // 1000    # 窗长，30ms
        self.win_shift = 10 * self.sample_freq // 1000  # 窗移，10ms
        self.except_frame_number_clean = 1 * 100              # 每句语音最长1秒，每秒100帧，一共100帧

        self.mix_dbs = mix_dbs
        self.is_train = is_train

        self.noise_path_list = self.read_path_list(noise_scp)
        self.noise_number = len(self.noise_path_list)
        # self.noise_list = []
        # for i in range(self.noise_number):
        #     sr, noise = wavfile.read(self.noise_path_list[i])
        #     self.noise_list.append(noise)

        self.speech_path_list = self.read_path_list(clean_scp)
        self.epoch = 0
        self.rescale_method = rs_method

    # @staticmethod
    # def read_sphere_wav(file_name):
    #     wav_file = open(file_name, 'rb')
    #     raw_header = wav_file.read(1024).decode('utf-8')
    #     raw_data = wav_file.read()
    #     wav_file.close()
    #     sample_count = len(raw_data) // 2
    #
    #     wav_data = np.zeros(shape=[sample_count], dtype=np.int32)
    #
    #     for i in range(sample_count):
    #         wav_data[i] = ctypes.c_int16(raw_data[2 * i + 1] << 8).value + ctypes.c_int16(raw_data[2 * i]).value
    #
    #     header_list = raw_header.split("\n")
    #     sphere_header = {}
    #     for s in header_list:
    #         if len(s) > 0 and s != "end_head":
    #             tmp = s.split(" ")
    #             if 0 < len(tmp) < 3:
    #                 sphere_header['Name'] = tmp[0]
    #             elif len(tmp[0]) > 0:
    #                 sphere_header[tmp[0]] = tmp[2]
    #
    #     return wav_data, sphere_header

    @staticmethod
    def read_wav(wav_path):
        #尽量用wav的文件吧，sph和wv1的文件可能有问题
        # if wav_path.endswith('wv1') or wav_path.endswith('sph'):
        #     data, header = FrameDataset.read_sphere_wav(wav_path)
        # else:
            # 这个地方读进来的数值是-32768~32767的
        data,sr = librosa.core.load(wav_path, sr=16000)
        return data


    # 这个提取过程是参考Kaldi的，你需要修改成你使用的分帧过程，分帧前做了预加重、加了随机白噪声
    # @staticmethod
    # def preprocess_speech(xx):
    #     pre_emphasis_weight = 0.97
    #     samples = xx.shape[0]
    #     # 预加重
    #     x = np.append(xx[0] - pre_emphasis_weight * xx[0], xx[1:] - pre_emphasis_weight * xx[:-1]).astype(np.float32)
    #     # 加入随机白噪声
    #     dither = np.random.standard_normal(samples)
    #     x += dither
    #     return x

    def enframe_speech_pair(self, _clean_speech, _noisy_speech, label):
        # 如果语音长于10秒就裁成10秒
        # if len(_clean_speech) > 10 * self.sample_freq:
        #     _clean_speech = _clean_speech[:10*self.sample_freq]
        #     _noisy_speech = _noisy_speech[:10*self.sample_freq]
        # _noisy_speech = self.preprocess_speech(_noisy_speech)
        # _clean_speech = self.preprocess_speech(_clean_speech)

        # 这个地方可以rescale一下wav sample的数据范围，文件里读出来的是-32768~32767的
        # 你可以添加你自己的方法，在这个地方实现，不要在别的地方，如果非要在别的地方，请务必要问问你老公。
        # if self.rescale_method is not None:
        #     if self.rescale_method.lower() == 'zhang':
        #         # 这是张老师给的方法，除以32768
        #         c = 32768.
        #     else:
        #         # 这是我的方法除以带噪语音的最大值
        #         c = np.max(np.abs(_noisy_speech))
        #     _noisy_speech /= c
        #     _clean_speech /= c

        frame_number = (len(_clean_speech) - self.win_len) // self.win_shift
        _clean_frames = np.zeros([self.except_frame_number_clean, self.win_len], np.float32)
        _noisy_frames = np.zeros([self.except_frame_number_clean, self.win_len], np.float32)
        for i in range(frame_number):
            _clean_frames[i, :] = _clean_speech[i * self.win_shift: i * self.win_shift + self.win_len]
            _noisy_frames[i, :] = _noisy_speech[i * self.win_shift: i * self.win_shift + self.win_len]
        frame_number = max((len(_clean_speech) - self.win_len) // self.win_shift, 100)
        return _clean_frames, _noisy_frames, frame_number, label

    # @staticmethod
    # def random_mix_speech_noise(clean_speech, noise, snr, noise_from, noise_to, is_norm=False):
    #     from numpy.linalg import norm
    #     to_mix_speech = clean_speech.astype(np.float32)
    #     to_mix_noise = np.array(noise[noise_from: noise_to])
    #     if len(clean_speech) < (noise_to - noise_from):
    #         noise_start = np.random.randint(noise_from, noise_to - len(clean_speech))
    #         to_mix_noise = noise[noise_start: noise_start + len(clean_speech)]
    #         to_mix_speech = clean_speech
    #     elif len(clean_speech) > (noise_to - noise_from):
    #         segs = len(clean_speech) // (noise_to - noise_from)
    #         to_mix_noise[:(segs - 1) * noise_to] = np.repeat(noise[noise_from: noise_to], segs)
    #         noise_start = np.random.randint(noise_from,
    #                                         noise_to - (len(clean_speech) - (segs - 1) * (noise_to - noise_from)))
    #         to_mix_noise[(segs - 1) * noise_to:] = noise[noise_start: noise_start + (
    #                     len(clean_speech) - (segs - 1) * (noise_to - noise_from))]
    #     to_mix_noise = to_mix_noise / norm(to_mix_noise) * norm(to_mix_speech) / np.sqrt(10.0 ** (0.1 * snr))
    #     check_snr = 10 * np.log10(np.square(norm(to_mix_speech) / norm(to_mix_noise)))
    #     if abs(check_snr - snr) > 1e-6:
    #         print("FATAL ERROR: snr calculate error!!!!")
    #     mix = to_mix_noise + to_mix_speech
    #     if is_norm:
    #         mix = mix - np.mean(mix)
    #         mix = mix / np.std(mix)
    #     return mix, to_mix_speech


    def __getitem__(self, index):
        # build my simulate utterance
        frame_list = {}
        idx = index
        clean_wav = self.read_wav(self.speech_path_list[idx])
        label = self.get_label(self.speech_path_list[idx])
        # snr = self.mix_dbs[np.random.randint(len(self.mix_dbs))]
        noise_wav = self.read_wav(self.noise_path_list[idx])
        # if self.is_train:
        #     noise_start = np.random.randint(0, len(noise_wav) // 2 - len(clean_wav))
        # else:
        #     noise_start = np.random.randint(len(noise_wav) // 2, len(noise_wav) - len(clean_wav))
        # _noisy_speech, _clean_speech = self.random_mix_speech_noise(clean_wav, noise_wav, snr, noise_start,
        #                                                        noise_start + len(clean_wav), False)
        frame_list['my_simu'] = self.enframe_speech_pair(clean_wav, noise_wav, label)

        return frame_list

    def __len__(self):
        return len(self.speech_path_list)


class FftDataloader(DataLoader):

    def __init__(self, dataset, opts, batch_size, shuffle=True, sampler=None, batch_sampler=None, num_workers=8,
                 collate_fn=None, pin_memory=False, drop_last=True, timeout=0, worker_init_fn=None):

        '''
        这个类生成一个mini-batch的特征。

        :param dataset:         传入上面那个FrameDataset的对象
        :param opts:            一个dict，opt['device']表示特征提取在cpu上做还是gpu上做，cpu上做则opt['device']='CPU'，GPU上做则opt['device']='cuda:0'
                                opt['win_len']表示窗长，多少个采样点
                                opt['win_type']表示窗类型，可以是'hanning', 'hamming', 'triangle', 'povey', 'blackman'
        :param batch_size:      每个batch多少句语音
        :param shuffle:         是否随机语音列表的顺序
        :param sampler:         默认即可
        :param batch_sampler:   默认即可
        :param num_workers:     后台并行的计算线程数，默认8，默认就行
        :param collate_fn:      计算一个batch特征的函数，传入my_collate_func，默认就行
        :param pin_memory:      默认
        :param drop_last:       当最后一个batch不足batch_size时，是否丢弃，默认丢弃，就默认吧
        :param timeout:         默认
        :param worker_init_fn:  默认

        使用示例：
        train_dataloader = FbankDataloader(train_dataset, opts, BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
        '''

        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, self.my_collate_func, pin_memory,
                         drop_last, timeout, worker_init_fn)
        self.opts = opts
        window = self.get_window(self.dataset.win_len, opts['win_type'])
        self.window = torch.Tensor(window[np.newaxis, np.newaxis, :]).to(self.opts['device'])
        self.next_power = np.int(np.power(2, np.ceil(np.log2(opts['win_len']))))

    # 提取能量谱
    def extract_pow(self, frames, frame_number):
        t_frames = torch.Tensor(frames).to(self.opts['device'])
        # 使每句话中帧的均值为0，这是kaldi的提取过程，你不需要的话可以注释这一行
        t_frames -= t_frames.mean(2, True)
        # 加窗
        t_frames = t_frames * self.window

        # 我做fft的时候是对2的整数次幂做的，如果你不需要的话，则注释下一行
        t_frames = F.pad(t_frames, (0, self.next_power - self.opts['win_len']))
        # 对每一帧做fft
        spect = torch.rfft(t_frames, 1)
        # 计算能量谱
        power_spect = torch.pow(spect[:, :, :, 0], 2.0) + torch.pow(spect[:, :, :, 1], 2.0)
        # c1 =torch.matmul(power_spect, self.melbank)
        return power_spect

    @staticmethod
    def get_window(win_len, win_type):
        # 计算各种各样的窗函数
        if win_type == 'hanning':
            win_len += 2
            window = np.hanning(win_len)
            window = window[1: -1]
        elif win_type == 'hamming':
            a = 2. * np.pi / (win_len - 1)
            window = 0.54 - 0.46 * np.cos(a * np.arange(win_len))
        elif win_type == 'triangle':
            window = 1. - (np.abs(win_len + 1. - 2. * np.arange(0., win_len + 2., 1.)) / (win_len + 1.))
            window = window[1: -1]
        elif win_type == 'povey':
            a = 2. * np.pi / (win_len - 1)
            window = np.power(0.5 - 0.5 * np.cos(np.arange(win_len) * a), 0.85)
        elif win_type == 'blackman':
            blackman_coeff = 0.42
            a = 2. * np.pi / (win_len - 1)
            window = blackman_coeff - 0.5 * np.cos(a * np.arange(win_len)) + \
                     (0.5 - blackman_coeff) * np.cos(2 * a * np.arange(win_len))
        else:
            window = np.ones(win_len)
        return window

    @staticmethod
    def my_collate_func(frames_list):
        batch_size = len(frames_list)
        batch_clean = np.zeros([batch_size, 100, 480], np.float32)
        batch_noisy = np.zeros([batch_size, 100, 480], np.float32)
        batch_mask = np.zeros([batch_size, 100], np.float32)
        batch_frame_number = [0] * len(frames_list) * 3
        batch_label = []
        i = 0
        for one_dict in frames_list:
            batch_clean[i, :, :] = one_dict['my_simu'][0]
            batch_noisy[i, :, :] = one_dict['my_simu'][1]
            batch_frame_number[i] = one_dict['my_simu'][2]
            batch_label[i] = one_dict['my_simu'][3]
            batch_mask[i, :batch_frame_number[i]] = 1.
            i += 1
        return batch_clean, batch_noisy, batch_frame_number, batch_label

    # 这个函数可以对齐带噪语音和纯净语音，你是直接合成的数据，不需要调用，不用管它。
    def calc_feats_and_align(self, clean_frames, noisy_frames, frame_number):
        feats = self.extract_pow(noisy_frames, frame_number)
        tgts = self.extract_pow(clean_frames, frame_number)
        # n, t, d = feats.size()
        # batch_size = n // 3
        # for i in range(2*batch_size, n):
        #     _idx = np.arange(frame_number[i])
        #     a = feats[i, :, :]
        #     b = tgts[i, :, :]
        #     norm_a = a.norm(dim=1)
        #     norm_b = b.norm(dim=1)
        #     max_coef = 0.
        #     max_idx = None
        #     for k in range(-3, 4):
        #         idx = (_idx + k) % frame_number[i]
        #         coeff = ((a[idx, :] * b[_idx, :]).sum(dim=1) / (norm_a[idx] * norm_b[_idx])).sum(dim=0)
        #         if coeff > max_coef:
        #             max_coef = coeff
        #             max_idx = idx
        #     feats[i, :frame_number[i], :] = a[max_idx, :]
        # feats.detach_()
        # tgts.detach_()
        return feats, tgts


