import torchvision.models as models
from torchvision.models.resnet import BasicBlock, ResNet

import torch


class CustomResnet(ResNet):
    def __init__(self):
        super(CustomResnet, self).__init__(BasicBlock, [2, 2, 2, 2])

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

