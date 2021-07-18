import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torchvision.models.resnet import resnet18, resnet34, resnet152
from utils import visualize_model
import wandb
from models.customResnet import CustomResnet
from train import Trainer
from torch.utils.data import DataLoader
from utils import imshow
from models.classifier import Net
from models.disciminator import Discriminator

print(torch.__version__)
plt.ion()  # interactive mode


def create_classifier_model(lr):
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    return net, criterion, optimizer


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

        classifier1, criterion1, optimizer1 = create_classifier_model(lr=config.learning_rate_classifier1)
        classifier2, criterion2, optimizer2 = create_classifier_model(lr=config.learning_rate_classifier2)

        discrimnator = Discriminator()

        trainer = Trainer(trainloader, testloader, [classifier1, classifier2], discrimnator)
        trainer.train_model([criterion1, criterion2], [optimizer1, optimizer2], num_epochs=config.epochs, checkpoint=None)
        trainer.validation()

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # # again no gradients needed
        # with torch.no_grad():
        #     for data in testloader:
        #         images, labels = data
        #         outputs = net(images)
        #         _, predictions = torch.max(outputs, 1)
        #         # collect the correct predictions for each class
        #         for label, prediction in zip(labels, predictions):
        #             if label == prediction:
        #                 correct_pred[classes[label]] += 1
        #             total_pred[classes[label]] += 1
        #
        # # print accuracy for each class
        # for classname, correct_count in correct_pred.items():
        #     accuracy = 100 * float(correct_count) / total_pred[classname]
        #     print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
        #                                                          accuracy))


def main():
    # args = parse_args()
    wandb.login()

    config = dict(
        epochs=4,
        batch_size=4,
        learning_rate_classifier1=0.001,
        learning_rate_classifier2=0.001,
        dataset="cifar10",
        num_classes=2,
        load=True,
        CUDA=True)

    model_pipeline(config)


if __name__ == '__main__':
    main()
