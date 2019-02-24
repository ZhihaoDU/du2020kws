from collections import ChainMap
import argparse
import os
import random
import sys

from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from openpyxl import load_workbook

import model_readlist as mod


# from .import model as mod
# import model as mod
# import os
#
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

class ConfigBuilder(object):
    def __init__(self, *default_configs):
        self.default_config = ChainMap(*default_configs)

    def build_argparse(self):
        parser = argparse.ArgumentParser()
        for key, value in self.default_config.items():
            key = "--{}".format(key)
            if isinstance(value, tuple):
                parser.add_argument(key, default=list(value), nargs=len(value), type=type(value[0]))
            elif isinstance(value, list):
                parser.add_argument(key, default=value, nargs="+", type=type(value[0]))
            elif isinstance(value, bool) and not value:
                parser.add_argument(key, action="store_true")
            else:
                parser.add_argument(key, default=value, type=type(value))
        return parser

    def config_from_argparse(self, parser=None):
        if not parser:
            parser = self.build_argparse()
        args = vars(parser.parse_known_args()[0])
        return ChainMap(args, self.default_config)


def print_eval(name, scores, labels, loss, end="\n"):
    batch_size = labels.size(0)
    accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).float().sum() / batch_size
    loss = loss.item()
    print("{} accuracy: {:>5}, loss: {:<25}".format(name, accuracy, loss), end=end)
    return accuracy


def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)


def evaluate(config, model=None, test_loader=None):
    print("testing")

    if not test_loader:
        _, _, test_set = mod.SpeechDataset.splits(config)
        test_loader = data.DataLoader(test_set, batch_size=128, shuffle=True)
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
    if not model:
        model = config["model_class"](config)
        model.load(config["input_file"])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()

    model.eval()
    print("cp0")
    criterion = nn.CrossEntropyLoss()
    results = []
    total = 0
    result_recall = []
    result_false = []
    print(len(test_loader))
    c = 0
    for model_in, labels in test_loader:
        model_in = Variable(model_in, requires_grad=False)
        if not config["no_cuda"]:
            model_in = model_in.cuda()
            labels = labels.cuda()
        scores = model(model_in)
        labels = Variable(labels, requires_grad=False)
        loss = criterion(scores, labels)
        results.append(print_eval("test", scores, labels, loss) * model_in.size(0))
        total += model_in.size(0)
        [recall, false] = eval_index_(labels, scores)
        result_recall.append(recall * model_in.size(0))
        result_false.append(false * model_in.size(0))
        c = c + 1
    print("c :{}".format(c))
    precision = sum(results) / total
    recall = sum(result_recall) / total
    false_alarm = sum(result_false) / total
    my_wb = load_workbook('/home/guyue/nfs_212/myBaseline/test.xlsx')
    activate_wb = my_wb.active
    # c = 0
    # row = '25'
    # for i in range(11):
    #     col_recall = chr(ord('D') + 2*c);
    #     sheet_index = ''.join([col_recall,row])
    #     activate_wb[sheet_index] = recall.numpy()[c]
    #     col_false = chr(ord('E') + 2*c )
    #     sheet_index = ''.join([col_false, row])
    #     activate_wb[sheet_index] = false_alarm.numpy()[c]
    #     c = c+1
    # activate_wb['Z25'] = recall.numpy()[11]
    # activate_wb['AA25'] = false_alarm.numpy()[11]
    # my_wb.save('/home/guyue/nfs_212/myBaseline/test.xlsx')
    print("final test accuracy: {}".format(precision))
    print("final test recall: {}".format(recall))
    print("final test false alarm: {}".format(false_alarm))


def eval_index_(labels, scores):
    batch_size = labels.size(0)
    pred = torch.max(scores, 1)[1].view(batch_size).data
    matrix = torch.zeros(12, 12)
    for i in range(batch_size):
        matrix[labels.data[i]][pred[i]] = matrix[labels.data[i]][pred[i]] + 1
    n = torch.zeros(12, 4)
    for i in range(12):
        for j in range(12):
            if i == j:
                n[i][0] = matrix[i][j]
            else:
                n[i][1] = n[i][1] + matrix[j][i]
                n[i][2] = n[i][2] + matrix[i][j]
    eval_index = torch.zeros(12, 2)
    for i in range(12):
        eval_index[i][0] = recall_denominator = n[i][2] + n[i][0]
        if recall_denominator != 0:
            eval_index[i][0] = n[i][0] / recall_denominator
        eval_index[i][1] = false_denominator = n[i][0] + n[i][1]
        if false_denominator != 0:
            eval_index[i][1] = n[i][1] / false_denominator
        print("label :{},Recall is: {},False alarm is :{}".format(i, eval_index[i][0], eval_index[i][1]))
    return [eval_index[:, 0], eval_index[:, 1]]

#--wanted_words yes no up down left right on off stop go --dev_every 1 --n_labels 12 --batch_size 10 --model cnn-trad-pool2 --input_file /home/guyue/nfs_212/mybaseline_cnn_trad_fpool2.pt --snr -3
def train(config):
    train_set, dev_set, test_set = mod.SpeechDataset.splits(config)
    model = config["model_class"](config)
    if config["input_file"]:
        model.load(config["input_file"])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    # optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][0], nesterov=config["use_nesterov"], weight_decay=config["weight_decay"], momentum=config["momentum"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    criterion = nn.CrossEntropyLoss()
    max_acc = 0

    train_loader = data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    dev_loader = data.DataLoader(dev_set, batch_size=min(len(dev_set), 16), shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=min(len(test_set), 16), shuffle=True)
    step_no = 0

    for epoch_idx in range(config["n_epochs"]):
        for batch_idx, (model_in, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            if not config["no_cuda"]:
                model_in = model_in.cuda()
                labels = labels.cuda()
            model_in = Variable(model_in, requires_grad=False)
            scores = model(model_in)
            labels = Variable(labels, requires_grad=False)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            step_no += 1
            if step_no > schedule_steps[sched_idx]:
                sched_idx += 1
                print("changing learning rate to {}".format(config["lr"][sched_idx]))
                # optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][sched_idx],nesterov=config["use_nesterov"], momentum=config["momentum"], weight_decay=config["weight_decay"])
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            print_eval("train step #{}".format(step_no), scores, labels, loss)

        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            model.eval()
            accs = []
            for model_in, labels in dev_loader:
                model_in = Variable(model_in, requires_grad=False)
                if not config["no_cuda"]:
                    model_in = model_in.cuda()
                    labels = labels.cuda()
                scores = model(model_in)
                labels = Variable(labels, requires_grad=False)
                loss = criterion(scores, labels)
                loss_numeric = loss.item()
                accs.append(print_eval("dev", scores, labels, loss))
            avg_acc = np.mean(accs)
            print("final dev accuracy: {}".format(avg_acc))
            if avg_acc > max_acc:
                print("saving best model...")
                max_acc = avg_acc
                model.save(config["output_file"])
    evaluate(config, model, test_loader)


# --wanted_words yes no up down left right on off stop go --dev_every 1 --n_labels 12 --output_file /home/guyue/nfs_212/mybaseline_cnn_trad_fpool.pt --batch_size 512
def recog_main():
    output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model", "model.pt")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[x.value for x in list(mod.ConfigType)], default="cnn-trad-pool2", type=str)
    config, _ = parser.parse_known_args()

    global_config = dict(no_cuda=False, n_epochs=50, lr=[1e-4], schedule=[np.inf], batch_size=256, dev_every=1, seed=0,
                         use_nesterov=False, input_file="", output_file=output_file, gpu_no=1, cache_size=32768,
                         momentum=0.9, weight_decay=0.00001, snr="-3")
    mod_cls = mod.find_model(config.model)
    builder = ConfigBuilder(
        mod.find_config(config.model),
        mod.SpeechDataset.default_config(),
        global_config)
    parser = builder.build_argparse()
    parser.add_argument("--mode", choices=["train", "eval"], default="eval", type=str)
    config = builder.config_from_argparse(parser)
    config["model_class"] = mod_cls
    set_seed(config)
    # if config["mode"] == "train":
    #     train(config)
    # elif config["mode"] == "eval":
    #     evaluate(config)
    return config


if __name__ == "__main__":
    recog_main()
