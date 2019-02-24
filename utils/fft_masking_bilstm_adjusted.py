import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
from kaldi_fft_dataset_adjusted import FftDataloader, FrameDataset
#from speech_utils import print_with_time
import hashlib

import numpy as np
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import librosa
import argparse
import model_readlist as mod
import train
from openpyxl import load_workbook

import random
from torch.autograd import Variable

import scipy.io as sio


class PowCRN(nn.Module):

    def __init__(self, conv_filters=32, lstm_hidden_units=32, noncausal=False):
        super(PowCRN, self).__init__()
        self.conv1 = nn.Sequential(
            # 1 x 101 x 241
            nn.Conv2d(1, conv_filters, 8, 4, 2),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
            # nf x 25 x 60
        )
        self.conv2 = nn.Sequential(
            # nf x 25 x 60
            nn.Conv2d(conv_filters, conv_filters, 8, 4, 2),
            nn.BatchNorm2d(conv_filters),
            nn.ReLU(),
            # nf x 6 x 15
        )
        self.lstm = nn.LSTM(conv_filters * 15, lstm_hidden_units, 1, batch_first=True, bidirectional=noncausal)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_units*2 if noncausal else lstm_hidden_units, conv_filters * 15),
            nn.BatchNorm1d(conv_filters * 15),
            nn.ReLU(),
        )
        self.deconv1 = nn.Sequential(
            # 2*nf x 6 x 15
            nn.ConvTranspose2d(conv_filters * 2, conv_filters, (9, 8), 4, 2),
            nn.BatchNorm2d(conv_filters),
            nn.LeakyReLU(0.1),
            # 2*nf x 25 x 60
        )
        self.deconv2 = nn.Sequential(
            # 2*nf x 25 x 60
            nn.ConvTranspose2d(conv_filters * 2, conv_filters, 9, 4, 2),
            nn.BatchNorm2d(conv_filters),
            nn.LeakyReLU(0.1),
            # nf x 101 x 241
            nn.Conv2d(conv_filters, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_filters = conv_filters
        self.lstm_hidden_units = lstm_hidden_units
        self.noncausal = noncausal

    def forward(self, input):
        c1 = self.conv1(input.unsqueeze(1))
        c2 = self.conv2(c1)
        lstm_in = c2.permute(0, 2, 1, 3).contiguous().view(-1, 6, self.conv_filters * 15)
        h0 = torch.zeros(2 if self.noncausal else 1, c2.size(0), self.lstm_hidden_units).to(input)
        c0 = torch.zeros(2 if self.noncausal else 1, c2.size(0), self.lstm_hidden_units).to(input)
        lstm_out, _ = self.lstm(lstm_in, (h0, c0))
        conv_in = self.fc(lstm_out.contiguous().view(-1, self.lstm_hidden_units*2 if self.noncausal else self.lstm_hidden_units))
        conv_in = conv_in.view(input.size(0), 6, self.conv_filters, 15).permute(0, 2, 1, 3)
        d1 = self.deconv1(torch.cat([conv_in, c2], dim=1))
        output = self.deconv2(torch.cat([d1, c1], dim=1))
        return output.squeeze(1)


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
        x = vutils.make_grid(feat[idx, :101, :].view(1,1,101,241).transpose(2,3), normalize=True, scale_each=True)
        writer.add_image("%s/noisy_mfb_%d" % (scope, idx), x, iteration)
        x = vutils.make_grid(label[idx, :101, :].view(1,1,101,241).transpose(2,3), normalize=True, scale_each=True)
        writer.add_image("%s/clean_mfb_%d" % (scope, idx), x, iteration)
        x = vutils.make_grid(predict[idx, :101, :].view(1,1,101,241).detach().transpose(2,3), normalize=True, scale_each=True)
        writer.add_image("%s/predict_%d" % (scope, idx), x, iteration)
        x = vutils.make_grid(mask[idx, :101, :].view(1,1,101,241).detach().transpose(2,3), normalize=True, scale_each=True)
        writer.add_image("%s/mask_%d" % (scope, idx), x, iteration)

def mfcc(feat,sr, n_mels, hop_length, n_fft, fmin, fmax, mel_basis, dct_filters):
    n_fft = 2 * (feat.shape[0] - 1)
    #mel_basis = torch.tensor(librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False,norm=1)).to('cuda:0')
    #mel_basis.requires_grad=True
    #dct_filters = torch.tensor(librosa.filters.dct(40, 40)).to('cuda:0')
    #dct_filters.requires_grad=True
    #mel_basis = librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False,norm=1)
    #return 0
    #return torch.matmul(mel_basis, feat)
    mel_spect = torch.matmul(mel_basis, torch.transpose(feat, 0,1))
    mel_spect[mel_spect > 0] = torch.log(mel_spect[mel_spect > 0])
    mfcc = torch.transpose(torch.matmul(dct_filters, mel_spect), 0, 1)
    mfcc = torch.transpose(mfcc.to(torch.float32), 0, 2)
    return mfcc 



if __name__ == '__main__':

    MODEL_NAME = "jointtraining_v1"
    BATCH_SIZE = 512
    TIME_STEPS = 1000
    FEAT_LENGTH = 320
    FRAME_LENGTH = 320 + (TIME_STEPS - 1) * 160
    FRAME_SHIFT = 16000 * 10

    print("|------------------------------------------------------------------|")
    print("|", ("Train %s: 2 layer, 384 units" % MODEL_NAME).center(64), "|")
    print("|------------------------------------------------------------------|")
    # print_with_time("Start to construct model...")
    print("Start to construct model...")
    TR05_NOISE_LIST = "/home/guyue/CNNProgram/datalist/enhancement/mixed_keyword_train.lst"
    TR05_CLEA_LIST = "/home/guyue/CNNProgram/datalist/enhancement/mapping_pure_keyword_train.lst"
    TR05_SLIENCE_LIST = "/home/guyue/CNNProgram/datalist/enhanced_into_baseline/keyword_slience_noise.lst"
    # TR05_ORGN_LIST = "wav_scp/tr05_orgn.scp"
    # TR05_SIMU_LIST = "wav_scp/tr05_simu.scp"
    # TR05_REAL_LIST = "wav_scp/tr05_real_noisy.scp"
    # TR05_CLOS_LIST = "wav_scp/tr05_real_close.scp"
   

   #test
    DT05_NOISE_LIST = "/home/guyue/CNNProgram/datalist/enhanced_into_baseline/snr-3_keyword_test.lst"
    DT05_CLEA_LIST =  "/home/guyue/CNNProgram/datalist/enhanced_into_baseline/mapping_pure_snr-3_keyword_test.lst"
    #DT05_NOISE_LIST = "/home/guyue/CNNProgram/datalist/enhancement/mixed_keyword_dev.lst"
    #DT05_CLEA_LIST = "/home/guyue/CNNProgram/datalist/enhancement/mapping_pure_keyword_dev.lst"
    DT05_SLIENCE_LIST = "/home/guyue/CNNProgram/datalist/enhanced_into_baseline/keyword_slience_noise.lst"
    # DT05_NOISE_LIST = "/home/guyue/CNNProgram/datalist/baseline/test.lst"
    # DT05_CLEA_LIST = "/home/guyue/CNNProgram/datalist/baseline/test.lst"
    # DT05_SLIENCE_LIST = "/home/guyue/CNNProgram/datalist/baseline/test.lst"
    # DT05_SIMU_LIST = "wav_scp/dt05_simu.scp"
    # DT05_ORGN_LIST = "wav_scp/dt05_orgn.scp"
    # DT05_REAL_LIST = "wav_scp/dt05_real_noisy.scp"
    # DT05_CLOS_LIST = "wav_scp/dt05_real_close.scp"

    opts = {}
    opts['win_len'] = 480
    opts['sr'] = 16000
    opts['device'] = torch.device('cuda:0')
    opts['mel_channels'] = 40
    opts['win_type'] = 'hann'


    #train_dataset = FrameDataset([0, 3, 6], TR05_NOISE_LIST, TR05_CLEA_LIST,TR05_SLIENCE_LIST, True, True)
    #train_dataloader = FftDataloader(train_dataset, opts, BATCH_SIZE, True, num_workers=8)
    valid_dataset = FrameDataset([0, 3, 6], DT05_NOISE_LIST, DT05_CLEA_LIST,DT05_SLIENCE_LIST, False, True)
    valid_dataloader = FftDataloader(valid_dataset, opts, BATCH_SIZE, True, num_workers=8)


    device = torch.device('cuda:0')
    # joint-training test
    #model = torch.load("/home/guyue/nfs_212/myjoint/joint_training/fft_masking_bilstm_adjusted_jointtraining_162/epoch_37.pkl")
    
    # enhancer
    model=torch.load("/home/guyue/joint_training/joint_model/enhancer_model/jointtraining_v1/epoch_1.pkl")
    #model = BiLSTM(241, 2, 384, 241).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    #if not os.path.exists("/home/guyue/joint_training/joint_model/enhancer_model/%s" % MODEL_NAME):
    #    os.mkdir("/home/guyue/joint_training/joint_model/enhancer_model/%s" % MODEL_NAME)

    last_loss = 0.
    best_loss = 10000.
    best_epoch = 0
    summary_count = 0
    print_interval = 10
    #writer = SummaryWriter("Tensorboard/%s/" % MODEL_NAME)
    # print_with_time("Model constructed, start to train the model.")
    print("Model constructed, start to train the model.")
    epoch = 1
    tr_global_step = 0
    dt_global_step = 0
    #torch.save(model, "/home/guyue/joint_training/joint_model/enhancer_model/%s/epoch_%d.pkl" % (MODEL_NAME, 0))
    mode = "test"
    if mode == 'train':
        config = train.recog_main()
        model_recog = config["model_class"](config)
        if config["input_file"]:
            model_recog.load(config["input_file"])
        # torch.cuda.set_device(config["gpu_no"])
        model_recog.cuda()
        optimizer_list = []
        #print(next(model.parameters()))
        optimizer_list.append(model.parameters())
        optimizer_list.append(model_recog.parameters())
        #mel_basis = torch.tensor(0)
        #dct_filters = torch.tensor(0)
        #mel_basis.requires_grad = True
        #dct_filters.requires_grad = True
        config["output_file"] = "/home/guyue/joint_training/joint_model/recog_model/epoch_%d.pt"%100
        model_recog.save(config["output_file"])
        mel_basis = torch.tensor(librosa.filters.mel(16000, 480, 40, 20, 4000, htk=False,norm=1)).to('cuda:0')
        mel_basis.requires_grad=True
        dct_filters = torch.tensor(librosa.filters.dct(40, 40)).to('cuda:0')
        dct_filters.requires_grad=True
        np.save('/home/guyue/joint_training/joint_model/mel_basis/epoch_%d.npy'%100,mel_basis.detach().cpu().numpy())
        np.save('/home/guyue/joint_training/joint_model/dct_filters/epoch_%d.npy'%100,dct_filters.detach().cpu().numpy())
        def optimizer_joint():
            enhancer_para = model.parameters()
            recog_para = model_recog.parameters()
            yield mel_basis
            yield dct_filters
            for i in enhancer_para:
                yield i
            for i in recog_para:
                yield i
        #optimizer_recog = torch.optim.Adam(model_recog.parameters(), lr=1e-4)
        optimizer_joint = torch.optim.Adam(optimizer_joint(), lr=1e-4)
        schedule_steps = config["schedule"]
        schedule_steps.append(np.inf)
        sched_idx = 0
        criterion = nn.CrossEntropyLoss()
        max_acc = 0

        step_no = 0

        while True:
            trained_utter_number = 0
            for iteration, (clean_frame, noisy_frame, frame_number, wav_label) in enumerate(train_dataloader):
                # calculate power spectrogram
                feat, label, irm = train_dataloader.calc_feats_and_align(clean_frame, noisy_frame, frame_number)
                trained_utter_number += feat.size(0)



                irm[torch.isinf(irm)] = 0.
                irm[torch.isnan(irm)] = 0.

                log_feat = torch.log(feat)
                log_feat[torch.isinf(log_feat)] = 0.
                log_feat[torch.isnan(log_feat)] = 0.

                log_label = torch.log(label)
                log_label[torch.isinf(log_label)] = 0.
                log_label[torch.isnan(log_label)] = 0.



                ideal = feat * irm
                log_ideal = torch.log(ideal)
                log_ideal[torch.isinf(log_ideal)] = 0.
                log_ideal[torch.isnan(log_ideal)] = 0.

                mask = model.forward(log_feat)
                predict = feat * mask

                log_predict = torch.log(predict)
                log_predict[torch.isinf(log_predict)] = 0.
                log_predict[torch.isnan(log_predict)] = 0.

                mse_loss = ((irm-mask)**2).mean()

                mfcc_orig = mfcc(torch.transpose(feat.to(torch.float64), 0, 2), 16000, 40, 160, 480, 20, 4000, mel_basis, dct_filters)
                mfcc_predict = mfcc(torch.transpose(predict.to(torch.float64), 0, 2), 16000, 40, 160, 480, 20, 4000,mel_basis, dct_filters)

                model_recog.train()
               # optimizer_joint.zero_grad()
                model_in_origin = Variable(mfcc_orig, requires_grad=False)
                model_in_predict = Variable(mfcc_predict, requires_grad=False)
                labels = torch.tensor(np.array(wav_label, dtype = int))
                if not config["no_cuda"]:
                    model_in_origin = model_in_origin.cuda()
                    model_in_predict = model_in_predict.cuda()
                    labels = labels.cuda()
                scores_origin = model_recog(model_in_origin)
                scores_predict = model_recog(model_in_predict)
                labels = Variable(labels, requires_grad=False)
                loss_origin = criterion(scores_origin, labels)
                loss_predict = criterion(scores_predict, labels)
                
                #joint_loss = mse_loss + loss_predict

                if step_no > schedule_steps[sched_idx]:
                    sched_idx += 1
                    print("changing learning rate to {}".format(config["lr"][sched_idx]))
                    # optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][sched_idx],nesterov=config["use_nesterov"], momentum=config["momentum"], weight_decay=config["weight_decay"])
                    optimizer_joint = torch.optim.Adam(test(), lr=1e-4)
                train.print_eval("ori train step #{}".format(step_no), scores_origin, labels, loss_origin)
                train.print_eval("pre train step #{}".format(step_no), scores_predict, labels, loss_predict)
                if tr_global_step % print_interval == 0:
                    # print_with_time("Epoch {}, Step {}, Utterance {}, Loss: {:.4f}".
                    #                 format(epoch, tr_global_step // print_interval, trained_utter_number, mse_loss.item()))
                    print("Epoch {}, Step {}, Utterance {}, Loss: {:.4f}".
                                    format(epoch, tr_global_step // print_interval, trained_utter_number, mse_loss.item()))
                    writer.add_scalar('train/recog_loss', loss_predict.item(), tr_global_step // print_interval)
                    writer.add_scalar('train/enhanced_loss', mse_loss.item(), tr_global_step // print_interval)
                    plot_image = vutils.make_grid(mel_basis.unsqueeze(0).unsqueeze(1).detach(), padding = 1, nrow = 1)
                    writer.add_image("train/mel_basis", plot_image,tr_global_step // print_interval)
                    plot_image = vutils.make_grid(dct_filters.unsqueeze(0).unsqueeze(1).detach(), padding=1, nrow=1)
                    writer.add_image("train/dct_filters", plot_image,tr_global_step // print_interval)  
                    writer.add_scalar('dct_filters_min',dct_filters.min().item(),tr_global_step // print_interval)
                    writer.add_scalar('dct_filters_mean',dct_filters.mean().item(),tr_global_step // print_interval)
                    writer.add_scalar('mel_basis_mean',mel_basis.mean().item(),tr_global_step // print_interval)
                    writer.add_scalar('mel_basis_min',mel_basis.min().item(),tr_global_step // print_interval)
                    writer.add_scalar('train/precision', train.print_eval("writer_add_acalar  #{}".format(step_no), scores_predict, labels, loss_predict),tr_global_step //print_interval)
                if tr_global_step % 100 == 0:
                    add_image_summary("train", log_feat, log_ideal, log_predict, mask, 3, BATCH_SIZE, tr_global_step // 100)
                optimizer_joint.zero_grad()
                # mse_loss.backward()
                # optimizer.step()
                loss_predict.backward()
                optimizer_joint.step()
                step_no += 1
                tr_global_step += 1
           # torch.save(model, "/home/guyue/nfs_212/myjoint/joint_training/%s/epoch_%d.pkl" % (MODEL_NAME, epoch))
            torch.save(model, "/home/guyue/joint_training/joint_model/enhancer_model/%s/epoch_%d.pkl" % (MODEL_NAME, epoch))
            np.save('/home/guyue/joint_training/joint_model/mel_basis/epoch_%d.npy'%epoch,mel_basis.detach().cpu().numpy())
            np.save('/home/guyue/joint_training/joint_model/dct_filters/epoch_%d.npy'%epoch,dct_filters.detach().cpu().numpy())
            config["output_file"] = "/home/guyue/joint_training/joint_model/recog_model/epoch_%d.pt"%epoch
            model_recog.save(config["output_file"])
            with torch.no_grad() :
                # print_with_time("Complete train %d epochs, start to evaluate performance in valid dataset." % epoch)
                print("Complete train %d epochs, start to evaluate performance in valid dataset." % epoch)
                valid_loss = 0.
                
                model_recog.eval()
                results_ori = []
                results_pre = []
                total = 0
                a = 0
                b = 0
                valid_loss = 0.
                valid_loss_predict = 0.
                valid_loss_enhanced = 0.
                c = 0
                result_recall_ori = []
                result_false_ori = []
                result_recall_pre = []
                result_false_pre = []

                for iteration, (clean_frame, noisy_frame, frame_number, wav_label) in enumerate(valid_dataloader):
                    feat, label, irm = valid_dataloader.calc_feats_and_align(clean_frame, noisy_frame, frame_number)
                    log_feat = torch.log(feat)
                    log_feat[torch.isinf(log_feat)] = 0.
                    log_feat[torch.isnan(log_feat)] = 0.

                    irm[torch.isinf(irm)] = 0.
                    irm[torch.isnan(irm)] = 0.

                    log_label = torch.log(label)
                    log_label[torch.isinf(log_label)] = 0.
                    log_label[torch.isnan(log_label)] = 0.

                    ideal = feat * irm
                    log_ideal = torch.log(ideal)
                    log_ideal[torch.isinf(log_ideal)] = 0.
                    log_ideal[torch.isnan(log_ideal)] = 0.

                    mask = model.forward(log_feat)
                    predict = feat * mask

                    log_predict = torch.log(predict)
                    log_predict[torch.isinf(log_predict)] = 0.
                    log_predict[torch.isnan(log_predict)] = 0.

                    mse_loss = ((irm-mask)**2).mean()
                    #valid_loss += mse_loss.item()
                    valid_loss_enhanced +=mse_loss.item()
                    
                    mfcc_orig = mfcc(torch.transpose(feat.to(torch.float64), 0,2), 16000, 40, 160, 480, 20,4000,mel_basis, dct_filters)
                    mfcc_predict = mfcc(torch.transpose(predict.to(torch.float64), 0, 2), 16000, 40, 160, 480, 20, 4000, mel_basis, dct_filters)

                    model_in_origin = Variable(mfcc_orig, requires_grad=False)
                    model_in_predict = Variable(mfcc_predict, requires_grad=False)
                    labels = torch.tensor(np.array(wav_label, dtype=int))
                    if not config["no_cuda"]:
                        model_in_origin = model_in_origin.cuda()
                        model_in_predict = model_in_predict.cuda()
                        labels = labels.cuda()
                    scores_origin = model_recog(model_in_origin)
                    scores_predict = model_recog(model_in_predict)
                    labels = Variable(labels, requires_grad=False)
                    loss_origin = criterion(scores_origin, labels)
                    loss_predict = criterion(scores_predict, labels)
                    valid_loss += loss_predict.item()
                    valid_loss_predict +=loss_predict.item()
                    
                    results_ori.append(
                        train.print_eval("test_orig", scores_origin, labels, loss_origin) * model_in_origin.size(0))
                    results_pre.append(
                        train.print_eval("test_predict", scores_predict, labels, loss_predict) * model_in_predict.size(
                            0))
                    total += model_in_origin.size(0)
                    [recall_ori, false_ori] = train.eval_index_(labels, scores_origin)
                    [recall_pre, false_pre] = train.eval_index_(labels, scores_predict)
                    result_recall_ori.append(recall_ori * model_in_origin.size(0))
                    result_false_ori.append(false_ori * model_in_origin.size(0))
                    result_recall_pre.append(recall_pre * model_in_predict.size(0))
                    result_false_pre.append(false_pre * model_in_predict.size(0))
                    precision_ori = sum(results_ori) / total
                    precision_pre = sum(results_pre) / total
                    print(['ddddd', precision_ori])
                    print(['eeeee', precision_pre])
                    c = c + 1

                    if dt_global_step % 100 == 0:
                        add_image_summary("valid", log_feat, log_ideal, log_predict, mask, 3, BATCH_SIZE, dt_global_step//100)
                    dt_global_step += 1
                    print("c :{}".format(c))
                precision_ori = sum(results_ori) / total
                precision_pre = sum(results_pre) / total
                recall_ori = sum(result_recall_ori) / total
                false_alarm_ori = sum(result_false_ori) / total
                recall_pre = sum(result_recall_pre) / total
                false_alarm_pre = sum(result_false_pre) / total

                avg_results_pre = np.mean(results_pre)
                print("final dev accuracy:{}".format(avg_results_pre))

                if avg_results_pre > max_acc:
                    print("saving best model...")
                    max_acc = avg_results_pre
                    config["output_file"] = "/home/guyue/joint_training/joint_model/recog_model/epoch_best.pt"
                    model_recog.save(config["output_file"])
                print("final test ori accuracy: {}".format(precision_ori))
                print("final test pre accuracy: {}".format(precision_pre))

                print("final test ori recall: {}".format(recall_ori))
                print("final test pre recall: {}".format(recall_pre))
                print("final test ori false alarm: {}".format(false_alarm_ori))
                print("final test pre false alarm: {}".format(false_alarm_pre))
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
            print("valid loss: %.4f, best loss: %.4f, best model:epoch_%d.pkl" %
                            (valid_loss, best_loss, best_epoch))
            writer.add_scalar("valid/valid_loss_enhanced", valid_loss_enhanced, epoch)
            writer.add_scalar("valid/valid_loss_predict", valid_loss_predict, epoch)
            writer.add_scalar("valid/precision", precision_pre, epoch)
            epoch += 1
        writer.close()
    else:
        with torch.no_grad():
            # print_with_time("Complete train %d epochs, start to evaluate performance in valid dataset." % epoch)
            print("Complete train %d epochs, start to evaluate performance in valid dataset." % epoch)

            config = train.recog_main()
            model_recog = config["model_class"](config)
            model_recog.load(config["input_file"])
            #torch.cuda.set_device(config["gpu_no"])
            model_recog.cuda()

            model_recog.eval()
            criterion = nn.CrossEntropyLoss()
            results_ori = []
            results_pre = []
            total = 0
            a = 0
            b = 0
            valid_loss = 0.
            c = 0
            result_recall_ori = []
            result_false_ori = []
            result_recall_pre = []
            result_false_pre = []
            for iteration, (clean_frame, noisy_frame, frame_number, wav_label) in enumerate(valid_dataloader):
                feat, label, irm = valid_dataloader.calc_feats_and_align(clean_frame, noisy_frame, frame_number)

                log_feat = torch.log(feat)
                log_feat[torch.isinf(log_feat)] = 0.
                log_feat[torch.isnan(log_feat)] = 0.

                irm[torch.isinf(irm)] = 0.
                irm[torch.isnan(irm)] = 0.

                log_label = torch.log(label)
                log_label[torch.isinf(log_label)] = 0.
                log_label[torch.isnan(log_label)] = 0.

                ideal = feat * irm
                log_ideal = torch.log(ideal)
                log_ideal[torch.isinf(log_ideal)] = 0.
                log_ideal[torch.isnan(log_ideal)] = 0.

                mask = model.forward(log_feat)
                predict = feat * mask

                log_predict = torch.log(predict)
                log_predict[torch.isinf(log_predict)] = 0.
                log_predict[torch.isnan(log_predict)] = 0.

                mse_loss = ((irm - mask) ** 2).mean()
                valid_loss += mse_loss.item()

                mfcc_orig = mfcc(torch.transpose(feat.to(torch.float64), 0, 2), 16000, 40, 160, 480, 20, 4000)
                mfcc_predict = mfcc(torch.transpose(predict.to(torch.float64), 0, 2), 16000, 40, 160, 480, 20, 4000)

                model_in_origin = Variable(mfcc_orig, requires_grad=False)
                model_in_predict = Variable(mfcc_predict, requires_grad=False)
                labels = torch.tensor(np.array(wav_label, dtype = int))
                if not config["no_cuda"]:
                    model_in_origin = model_in_origin.cuda()
                    model_in_predict = model_in_predict.cuda()
                    labels = labels.cuda()
                scores_origin = model_recog(model_in_origin)
                scores_predict = model_recog(model_in_predict)
                labels = Variable(labels, requires_grad=False)
                loss_origin = criterion(scores_origin, labels)
                loss_predict = criterion(scores_predict, labels)
                results_ori.append(train.print_eval("test_orig", scores_origin, labels, loss_origin) * model_in_origin.size(0))
                results_pre.append(train.print_eval("test_predict", scores_predict, labels, loss_predict) * model_in_predict.size(0))
                total += model_in_origin.size(0)
                [recall_ori, false_ori] = train.eval_index_(labels, scores_origin)
                [recall_pre, false_pre] = train.eval_index_(labels, scores_predict)
                result_recall_ori.append(recall_ori * model_in_origin.size(0))
                result_false_ori.append(false_ori * model_in_origin.size(0))
                result_recall_pre.append(recall_pre * model_in_predict.size(0))
                result_false_pre.append(false_pre * model_in_predict.size(0))
                precision_ori = sum(results_ori) / total
                precision_pre = sum(results_pre) / total
                print(['ddddd',precision_ori])
                print(['eeeee', precision_pre])
                c = c+1
                if dt_global_step % 100 == 0:
                    add_image_summary("valid", log_feat, log_ideal, log_predict, mask, 3, BATCH_SIZE,
                                      dt_global_step // 100)
                dt_global_step += 1
                print("c :{}".format(c))
            precision_ori = sum(results_ori) / total
            precision_pre = sum(results_pre) / total
            recall_ori = sum(result_recall_ori) / total
            false_alarm_ori = sum(result_false_ori) / total
            recall_pre = sum(result_recall_pre) / total
            false_alarm_pre = sum(result_false_pre) / total
            print("final test ori accuracy: {}".format(precision_ori))
            print("final test pre accuracy: {}".format(precision_pre))

            print("final test ori recall: {}".format(recall_ori))
            print("final test pre recall: {}".format(recall_pre))
            print("final test ori false alarm: {}".format(false_alarm_ori))
            print("final test pre false alarm: {}".format(false_alarm_pre))
            my_wb = load_workbook('/home/guyue/kws_result/kws_result.xlsx')
            activate_wb = my_wb.active
            c = 0
            row_ori = '16'
            row_pre = '4'
            for i in range(11):
                col_recall = chr(ord('D') + 2*c)
                sheet_index = ''.join([col_recall,row_ori])
                activate_wb[sheet_index] = recall_ori.numpy()[c]
                sheet_index = ''.join([col_recall, row_pre])
                activate_wb[sheet_index] = recall_pre.numpy()[c]
                col_false = chr(ord('E') + 2*c )
                sheet_index = ''.join([col_false, row_ori])
                activate_wb[sheet_index] = false_alarm_ori.numpy()[c]
                sheet_index = ''.join([col_false, row_pre])
                activate_wb[sheet_index] = false_alarm_pre.numpy()[c]
                c = c+1

            activate_wb['Z'+ row_ori] = recall_ori.numpy()[11]
            activate_wb['AA'+ row_ori] = false_alarm_ori.numpy()[11]
            activate_wb['Z'+ row_pre] = recall_pre.numpy()[11]
            activate_wb['AA'+ row_pre] = false_alarm_pre.numpy()[11]

            # activate_wb['C'+ row_ori] = precision_ori.cpu().numpy()[0]
            # activate_wb['C'+row_pre] = precision_pre.cpu().numpy()[0]
            #my_wb.save('/home/guyue/nfs_212/myBaseline/result.xlsx')

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
        # print_with_time("valid loss: %.4f, best loss: %.4f, best model:epoch_%d.pkl" %
        #                 (valid_loss, best_loss, best_epoch))
        print("valid loss: %.4f, best loss: %.4f, best model:epoch_%d.pkl" %
              (valid_loss/dt_global_step, best_loss, best_epoch))
        #writer.add_scalar("valid/mse_loss", valid_loss, epoch)
