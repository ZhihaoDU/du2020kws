import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
from kaldi_fft_dataset import FftDataloader, FrameDataset
#from speech_utils import print_with_time
import hashlib

import numpy as np
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import scipy.io as sio

class BiLSTM(nn.Module):

    def __init__(self, input_size, layer_number, hidden_units, out_dim):
        super(BiLSTM, self).__init__()
        self.layer_number = layer_number
        self.hidden_units = hidden_units
        self.out_dim = out_dim
        self.lstm = nn.LSTM(input_size, hidden_units, layer_number, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_units*2, out_dim)
        self.device = torch.device('cuda')

    def forward(self, x):
        h0 = torch.zeros(self.layer_number*2, x.size(0), self.hidden_units).to(self.device)
        c0 = torch.zeros(self.layer_number*2, x.size(0), self.hidden_units).to(self.device)
        out, _ = self.lstm(x, (h0,c0))
        seq_len = out.shape[1]
        out = out.contiguous().view([-1, self.hidden_units*2])
        out = torch.sigmoid(self.fc(out))
        out = out.contiguous().view([-1, seq_len, self.out_dim])
        return out

def add_image_summary(scope, feat, label, predict, mask, image_num, batch_size, iteration):
    feat_dim = feat.size(2)
    # image_num should be smaller than batch_size
    for i in range(image_num):
        idx = i
        x = vutils.make_grid(feat[idx, :100, :].permute([1, 0]).contiguous().view(feat_dim, 100), normalize=True, scale_each=True)
        writer.add_image("%s/noisy_mfb_%d" % (scope, idx), x, iteration)
        x = vutils.make_grid(label[idx, :100, :].permute([1, 0]).contiguous().view(feat_dim, 100), normalize=True, scale_each=True)
        writer.add_image("%s/clean_mfb_%d" % (scope, idx), x, iteration)
        x = vutils.make_grid(predict[idx, :100, :].permute([1, 0]).contiguous().view(feat_dim, 100), normalize=True, scale_each=True)
        writer.add_image("%s/predict_%d" % (scope, idx), x, iteration)
        x = vutils.make_grid(mask[idx, :100, :].permute([1, 0]).contiguous().view(feat_dim, 100), normalize=True, scale_each=True)
        writer.add_image("%s/mask_%d" % (scope, idx), x, iteration)

if __name__ == '__main__':

    MODEL_NAME = "fft_masking_bilstm_3"
    BATCH_SIZE = 2
    TIME_STEPS = 1000
    FEAT_LENGTH = 320
    FRAME_LENGTH = 320 + (TIME_STEPS - 1) * 160
    FRAME_SHIFT = 16000 * 10

    print("|------------------------------------------------------------------|")
    print("|", ("Train %s: 2 layer, 384 units" % MODEL_NAME).center(64), "|")
    print("|------------------------------------------------------------------|")
    # print_with_time("Start to construct model...")
    print("Start to construct model...")
    TR05_NOISE_LIST = "/home/guyue/CNNProgram/datalist/snr_debug.lst"
    TR05_CLEA_LIST = "/home/guyue/CNNProgram/datalist/snr_debug.lst"
    # TR05_NOISE_LIST = "/home/guyue/CNNProgram/datalist/mixed_all_train.lst"
    # TR05_CLEA_LIST = "/home/guyue/CNNProgram/datalist/mapping_pure_train.lst"
    # TR05_ORGN_LIST = "wav_scp/tr05_orgn.scp"
    # TR05_SIMU_LIST = "wav_scp/tr05_simu.scp"
    # TR05_REAL_LIST = "wav_scp/tr05_real_noisy.scp"
    # TR05_CLOS_LIST = "wav_scp/tr05_real_close.scp"

    DT05_NOISE_LIST = "/home/guyue/CNNProgram/datalist/snr_debug.lst"
    DT05_CLEA_LIST = "/home/guyue/CNNProgram/datalist/snr_debug.lst"
    # DT05_NOISE_LIST = "/home/guyue/CNNProgram/datalist/mixed_all_dev.lst"
    # DT05_CLEA_LIST = "/home/guyue/CNNProgram/datalist/mapping_pure_dev.lst"
    # DT05_SIMU_LIST = "wav_scp/dt05_simu.scp"
    # DT05_ORGN_LIST = "wav_scp/dt05_orgn.scp"
    # DT05_REAL_LIST = "wav_scp/dt05_real_noisy.scp"
    # DT05_CLOS_LIST = "wav_scp/dt05_real_close.scp"

    opts = {}
    opts['win_len'] = 480
    opts['sr'] = 16000
    opts['device'] = torch.device('cuda:0')
    opts['mel_channels'] = 40
    opts['win_type'] = 'hamming'


    train_dataset = FrameDataset([0, 3, 6], TR05_NOISE_LIST, TR05_CLEA_LIST, True, True)
    train_dataloader = FftDataloader(train_dataset, opts, BATCH_SIZE, True, num_workers=8)
    valid_dataset = FrameDataset([0, 3, 6], DT05_NOISE_LIST, DT05_CLEA_LIST, False, True)
    valid_dataloader = FftDataloader(valid_dataset, opts, BATCH_SIZE, True, num_workers=8)


    device = torch.device('cuda:0')
    model=torch.load("/home/guyue/nfs_212/myEnhancement/fft_masking_bilstm_2/epoch_9.pkl")
    #model = BiLSTM(257, 2, 384, 257).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if not os.path.exists("/home/guyue/nfs_212/myEnhancement/%s" % MODEL_NAME):
        os.mkdir("/home/guyue/nfs_212/myEnhancement/%s" % MODEL_NAME)

    last_loss = 0.
    best_loss = 10000.
    best_epoch = 0
    summary_count = 0
    print_interval = 10
    writer = SummaryWriter("Tensorboard/%s/" % MODEL_NAME)
    # print_with_time("Model constructed, start to train the model.")
    print("Model constructed, start to train the model.")
    epoch = 11
    tr_global_step = 0
    dt_global_step = 0
    torch.save(model, "/home/guyue/nfs_212/myEnhancement/%s/epoch_test_%d.pkl" % (MODEL_NAME, 10))
    while True:
        trained_utter_number = 0
        for iteration, (clean_frame, noisy_frame, frame_number) in enumerate(train_dataloader):
            # calculate power spectrogram
            feat, label = train_dataloader.calc_feats_and_align(clean_frame, noisy_frame, frame_number)
            trained_utter_number += feat.size(0)

            log_feat = torch.log(feat)
            log_feat[torch.isinf(log_feat)] = 0.
            log_feat[torch.isnan(log_feat)] = 0.

            log_label = torch.log(label)
            log_label[torch.isinf(log_label)] = 0.
            log_label[torch.isnan(log_label)] = 0.

            irm = label / feat
            irm[torch.isnan(irm)] = 0.
            irm[torch.isinf(irm)] = 0.
            irm[irm > 1.] = 1.
            irm[irm < 0.] = 0.
            ideal = feat * irm

            mask = model.forward(log_feat)
            predict = feat * mask

            log_predict = torch.log(predict)
            log_predict[torch.isinf(log_predict)] = 0.
            log_predict[torch.isnan(log_predict)] = 0.

            mse_loss = ((irm-mask)**2).mean()

            if tr_global_step % print_interval == 0:
                # print_with_time("Epoch {}, Step {}, Utterance {}, Loss: {:.4f}".
                #                 format(epoch, tr_global_step // print_interval, trained_utter_number, mse_loss.item()))
                print("Epoch {}, Step {}, Utterance {}, Loss: {:.4f}".
                                format(epoch, tr_global_step // print_interval, trained_utter_number, mse_loss.item()))
                writer.add_scalar('train/mse_loss', mse_loss.item(), tr_global_step // print_interval)
            if tr_global_step % 100 == 0:
                add_image_summary("train", log_feat, log_label, log_predict, mask, 3, BATCH_SIZE, tr_global_step // 100)
            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()
            tr_global_step += 1
        torch.save(model, "/home/guyue/nfs_212/myEnhancement/%s/epoch_%d.pkl" % (MODEL_NAME, epoch))

        with torch.no_grad() :
            # print_with_time("Complete train %d epochs, start to evaluate performance in valid dataset." % epoch)
            print("Complete train %d epochs, start to evaluate performance in valid dataset." % epoch)
            valid_loss = 0.
            for iteration, (clean_frame, noisy_frame, frame_number) in enumerate(valid_dataloader):
                feat, label = valid_dataloader.calc_feats_and_align(clean_frame, noisy_frame, frame_number)
                log_feat = torch.log(feat)
                log_feat[torch.isinf(log_feat)] = 0.
                log_feat[torch.isnan(log_feat)] = 0.

                log_label = torch.log(label)
                log_label[torch.isinf(log_label)] = 0.
                log_label[torch.isnan(log_label)] = 0.

                irm = label / feat
                irm[torch.isnan(irm)] = 0.
                irm[torch.isinf(irm)] = 0.
                irm[irm > 1.] = 1.
                irm[irm < 0.] = 0.
                ideal = feat * irm

                mask = model.forward(log_feat)
                predict = feat * mask

                log_predict = torch.log(predict)
                log_predict[torch.isinf(log_predict)] = 0.
                log_predict[torch.isnan(log_predict)] = 0.

                mse_loss = ((irm-mask)**2).mean()
                valid_loss += mse_loss.item()
                if dt_global_step % 100 == 0:
                    add_image_summary("valid", log_feat, log_label, log_predict, mask, 3, BATCH_SIZE, dt_global_step//100)
                dt_global_step += 1
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
        # print_with_time("valid loss: %.4f, best loss: %.4f, best model:epoch_%d.pkl" %
        #                 (valid_loss, best_loss, best_epoch))
        print("valid loss: %.4f, best loss: %.4f, best model:epoch_%d.pkl" %
                        (valid_loss, best_loss, best_epoch))
        writer.add_scalar("valid/mse_loss", valid_loss, epoch)
        epoch += 1
    writer.close()
