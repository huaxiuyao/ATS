import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Conv_Standard_ANIL(nn.Module):
    def __init__(self, args, x_dim, hid_dim, z_dim, final_layer_size, stride=1):
        super(Conv_Standard_ANIL, self).__init__()
        self.args = args
        self.stride = stride
        self.net = nn.Sequential(self.conv_block(x_dim, hid_dim), self.conv_block(hid_dim, hid_dim),
                                 self.conv_block(hid_dim, hid_dim), self.conv_block(hid_dim, z_dim), Flatten())
        self.dist = Beta(torch.FloatTensor([2]), torch.FloatTensor([2]))
        self.hid_dim = hid_dim

        self.logits = nn.Linear(final_layer_size, self.args.num_classes)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=self.stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def functional_conv_block(self, x, weights, biases,
                              bn_weights, bn_biases, is_training):
        """Performs 3x3 convolution, ReLu activation, 2x2 max pooling in a functional fashion.

        # Arguments:
            x: Input Tensor for the conv block
            weights: Weights for the convolutional block
            biases: Biases for the convolutional block
            bn_weights:
            bn_biases:
        """
        x = F.conv2d(x, weights, biases, padding=1, stride=self.stride)
        x = F.batch_norm(x, running_mean=None, running_var=None, weight=bn_weights, bias=bn_biases,
                         training=is_training)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        return x

    def forward(self, x):
        x = self.net(x)

        x = x.view(x.size(0), -1)

        return self.logits(x)

    def functional_forward(self, x, weights, is_training=True):
        x = self.net(x)

        x = x.view(x.size(0), -1)

        x = F.linear(x, weights['weight'], weights['bias'])

        return x

    def functional_forward_val(self, x, weights, weights_logits, is_training=True):
        for block in range(4):
            x = self.functional_conv_block(x, weights[f'net.{block}.0.weight'], weights[f'net.{block}.0.bias'],
                                           weights.get(f'net.{block}.1.weight'), weights.get(f'net.{block}.1.bias'),
                                           is_training)

        x = x.view(x.size(0), -1)

        x = F.linear(x, weights_logits['weight'], weights_logits['bias'])

        return x
