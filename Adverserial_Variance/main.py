import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms

from models.classifier import Net
from models.disciminator import Discriminator
from models.resnet import ResNet18
from train import Trainer

print(torch.__version__)
plt.ion()  # interactive mode


def create_classifier_model(lr, model='basic'):
    if model == 'resnet':
        net = ResNet18()
    else:
        net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), weight_decay=0.0001, lr=lr)
    return net, criterion, optimizer


def create_discriminator():
    discriminator = Discriminator()
    optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterionD = nn.BCELoss()
    return discriminator, optimizerD, criterionD


def model_pipeline(hyperparameters):
    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                                  shuffle=True)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                 shuffle=False)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        classifier1, criterion1, optimizer1 = create_classifier_model(lr=config.learning_rate_classifier1,
                                                                      )
        classifier2, criterion2, optimizer2 = create_classifier_model(lr=config.learning_rate_classifier2,
                                                                      )

        discriminator, optimizerD, criterionD = create_discriminator()

        trainer = Trainer(trainloader, testloader, [classifier1, classifier2], discriminator)
        trainer.train_model([criterion1, criterion2], [optimizer1, optimizer2], optimizerD, criterionD, num_epochs=config.epochs,
                            checkpoint=None)
        trainer.validation()

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}


def main():
    # args = parse_args()
    wandb.login()

    config = dict(
        epochs=2,
        batch_size=32,
        learning_rate_classifier1=0.01,
        learning_rate_classifier2=0.01,
        dataset="cifar10",
        num_classes=2,
        load=True,
        CUDA=True)

    model_pipeline(config)


if __name__ == '__main__':
    main()
