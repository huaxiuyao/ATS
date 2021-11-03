import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class FCNet(nn.Module):
    def __init__(self, args, x_dim, hid_dim):
        super(FCNet, self).__init__()
        self.args = args
        self.net = nn.Sequential(self.fc_block(x_dim, hid_dim), self.fc_block(hid_dim, hid_dim))
        self.hid_dim = hid_dim

        self.logits = nn.Linear(hid_dim, 1)
        self.leaky_relu = torch.nn.LeakyReLU()

    def fc_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )

    def functional_conv_block(self, x, weights, biases,
                              bn_weights, bn_biases, is_training):
        x = F.linear(x, weights, biases)
        x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases,
                         training=is_training)
        x = F.leaky_relu(x)
        return x

    def forward(self, x):
        x = self.net(x)

        x = x.view(x.size(0), -1)

        return self.leaky_relu(self.logits(x))

    def functional_forward(self, x, weights, is_training=True):
        x = self.net(x)

        x = x.view(x.size(0), -1)

        x = F.leaky_relu(F.linear(x, weights['weight'], weights['bias']))

        return x

    def functional_forward_val(self, x, weights, weights_logits, is_training=True):
        for block in range(2):
            x = self.functional_conv_block(x, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'],
                                           weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'),
                                           is_training)
        x = x.view(x.size(0), -1)

        x = F.leaky_relu(F.linear(x, weights_logits['weight'], weights_logits['bias']))

        return x
