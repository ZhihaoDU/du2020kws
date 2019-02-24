import torch
import torch.nn as nn
import torch.nn.functional as F
from model_readlist import SpeechModel, _configs


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


def count_parameters(model):
    parameter_num_list = [p.numel() for p in model.parameters() if p.requires_grad]
    return sum(parameter_num_list), parameter_num_list


if __name__ == '__main__':
    # model = SpeechModel(_configs["cnn-trad-pool2"])
    model = PowCRN(16, 32, True)
    input = torch.randn(2, 101, 241)
    output = model.forward(input)
    print(output.size())
    total_number, each_layer_number = count_parameters(model)
    print(each_layer_number)
    print(total_number)
