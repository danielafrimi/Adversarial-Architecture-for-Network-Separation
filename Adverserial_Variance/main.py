import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms

from models.classifier import Net
from models.disciminator import Discriminator, SiameseDiscriminator
from models.resnet import ResNet18
from train import Trainer
from utils import extract_images_to_dataset

print(torch.__version__)
plt.ion()  # interactive mode


def create_classifier_model(lr, num_classes, model='basic'):
    if model == 'resnet':
        net = ResNet18()
    else:
        net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), weight_decay=0.0001, lr=lr)
    return net, criterion, optimizer


def create_discriminator():
    discriminator = SiameseDiscriminator()
    optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterionD = nn.BCELoss()
    return discriminator, optimizerD, criterionD


def model_pipeline(hyperparameters):
    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # Data augmentation and normalization for training
        # Just normalization for validation
        transform = {
            'train': transforms.Compose([
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        cat_dog_trainset, cat_dog_testset = extract_images_to_dataset()

        # Create datasetLoaders from trainset and testset
        trainsetLoader = DataLoader(cat_dog_trainset, batch_size=16, shuffle=True)
        testsetLoader = DataLoader(cat_dog_testset, batch_size=16, shuffle=False)

        # todo this is on the whole dataset
        # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
        #                                         download=False, transform=transform['train'])
        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
        #                                           shuffle=True)
        #
        # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
        #                                        download=False, transform=transform['test'])
        # testloader = torch.utils.data.DataLoader(testset, batch_size=1,
        #                                          shuffle=False)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        classifier1, criterion1, optimizer1 = create_classifier_model(lr=config.learning_rate_classifier1,
                                                                      num_classes=config.num_classes, model='resnet')
        classifier2, criterion2, optimizer2 = create_classifier_model(lr=config.learning_rate_classifier2,
                                                                      num_classes=config.num_classes, model='resnet')

        discriminator, optimizerD, criterionD = create_discriminator()

        trainer = Trainer(trainsetLoader, testsetLoader, [classifier1, classifier2], discriminator)
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
        epochs=1,
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
