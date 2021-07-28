import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms

from models.classifier import Net
from models.disciminator import SiameseDiscriminator
from models.resnet import ResNet18
from train import Trainer
from utils import extract_images_to_dataset, get_trainloader_all_cifar10

print(torch.__version__)
plt.ion()  # interactive mode


def create_classifier_model(lr, num_classes, model='basic'):
    if model == 'resnet':
        net = ResNet18()
    else:
        net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return net, criterion, optimizer, scheduler


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
        trainsetLoader = DataLoader(cat_dog_trainset, batch_size=config.batch_size, shuffle=True)
        testsetLoader = DataLoader(cat_dog_testset, batch_size=config.batch_size, shuffle=False)

        trainsetLoader, testsetLoader = get_trainloader_all_cifar10()

        classifier1, criterion1, optimizer1, scheduler1 = create_classifier_model(lr=config.learning_rate_classifier1,
                                                                                  num_classes=config.num_classes,
                                                                                  model='resnet')
        classifier2, criterion2, optimizer2, scheduler2 = create_classifier_model(lr=config.learning_rate_classifier2,
                                                                                  num_classes=config.num_classes,
                                                                                  model='resnet')

        discriminator, optimizerD, criterionD = create_discriminator()

        trainer = Trainer(trainsetLoader, testsetLoader, [classifier1, classifier2], discriminator)
        trainer.train_model([criterion1, criterion2], [optimizer1, optimizer2], [scheduler1, scheduler2], optimizerD,
                            criterionD, num_epochs=config.epochs, checkpoint=None)

        trainer.validation()


def main():
    # args = parse_args()
    wandb.login()

    config = dict(
        epochs=20,
        batch_size=64,
        learning_rate_classifier1=0.01,
        learning_rate_classifier2=0.01,
        dataset="cifar10",
        num_classes=2)

    model_pipeline(config)


if __name__ == '__main__':
    main()
