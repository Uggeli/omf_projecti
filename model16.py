import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent(nn.Module):
    def __init__(self, in_c, n_classes, name, training_days, device='cuda'):
        super(Agent, self).__init__()
        self.device = device
        self.org_name = name
        self.name = name
        self.mutations = 0
        self.total_rewards = []

        self.enc_sizes = [in_c] + [128 for _ in range(36)]

        conv_blocks = [self.conv_block(in_f, out_f, kernel_size=3, stride=2, padding=1)
                       for in_f, out_f in zip(self.enc_sizes,
                       self.enc_sizes[1:])]

        self.encoder = nn.Sequential(*conv_blocks)

        self.decoder = nn.Sequential(
            nn.Linear(9052160, int(9052160 / 2)),
            nn.Sigmoid(),
            nn.Linear(int(9052160 / 2), n_classes)
        )

        self.maxpool = nn.MaxPool1d(5)

        self.dropout = nn.Dropout()

        self.lstm = nn.LSTM(training_days, 128, num_layers=(len(self.encoder)))
        self.init_hidden(in_c)

    def conv_block(self, in_f, out_f, *args, **kwargs):
        return nn.Sequential(
            nn.Conv1d(in_f, out_f, *args, **kwargs),
            nn.BatchNorm1d(out_f),
            nn.ReLU()
        )

    def forward(self, data_in):
        encoder_out = self.encoder(data_in)
        x = self.maxpool(encoder_out)

        y, (self.h_n, self.c_n) = self.lstm(data_in.view(320, 220, 4),
                                            (self.h_n.detach(),
                                             self.c_n.detach()))

        out = torch.cat([y.view(-1, 128, 1), x.view(-1, 128, 1)])
        testi = out.view(x.size(0), -1)
        flat = torch.flatten(out)
        out = self.decoder(out.view(x.size(0), -1))
        # out2 = self.decoder(out)
        return out

    def init_hidden(self, in_c):
        self.h_n = torch.zeros(len(self.encoder), 220, 128).to(self.device)
        self.c_n = torch.zeros(len(self.encoder), 220, 128).to(self.device)
