"""
The discriminator is made up of strided convolution layers, batch norm layers, and LeakyReLU activations.
 The input is a 3x64x64 input image and the output is a scalar probability that the input is from the
 real data distribution.
"""
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseDiscriminator(nn.Module):

    def __init__(self):
        super(SiameseDiscriminator, self).__init__()
        # TODO replace pooling layer with stride conv
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),  # 128@42*42
            nn.MaxPool2d(2),  # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(),  # 128@18*18
            nn.MaxPool2d(2),  # 128@9*9
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),  # 256@6*6
        )
        self.liner = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)

        self.apply(init_weights)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        """

        :param x1: feature map of classifier1
        :param x2: feature map of classifier2
        :return: Score that indicates of similarity (1 is high)
        """
        print(x1.shape)
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out


# todo  both are Adam optimizers with learning rate 0.0002 and Beta1 = 0.5.
# custom weights initialization called on netD
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, ngpu=0, nc=3, ndf=64):
        """
        :param ngpu: number of GPUs available. If this is 0, code will run in CPU mode.
         If this number is greater than 0 it will run on that number of GPUs
        :param nc: number of color channels in the input images. For color images this is 3
        :param ndf: sets the depth of feature maps propagated through the discriminator
        """
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.apply(init_weights)

    def forward(self, input):
        return self.main(input)
