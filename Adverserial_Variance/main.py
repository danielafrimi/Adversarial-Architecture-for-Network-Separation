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
from models.CustomResnet import CustomResnet
from train import Trainer
from torch.utils.data import DataLoader
from utils import get_dataset, imshow

print(torch.__version__)
plt.ion()  # interactive mode


def create_model(config, device):
    # todo !!
    resnet_model = CustomResnet()
    resnet_model.load_state_dict(resnet18(pretrained=False).state_dict())

    # # Parameters of newly constructed modules have requires_grad=True by default
    # for param in resnet_model.parameters():
    #     param.requires_grad = False

    # Change the prediction for 2 classes
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, config.num_classes)

    resnet_model = resnet_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(resnet_model.parameters(), lr=config.learning_rate_classifier1, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    return resnet_model, criterion, optimizer_conv, exp_lr_scheduler


def model_pipeline(hyperparameters):
    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        data_dir = 'data'
        CHECK_POINT_PATH = 'checkpoint.tar'
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                          for x in ['train', 'val']}

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
                       for x in ['train', 'val']}

        cat_dog_trainset, cat_dog_testset = get_dataset()

        # Create datasetLoaders from trainset and testset
        trainsetLoader = DataLoader(cat_dog_trainset, batch_size=16, shuffle=True)
        testsetLoader = DataLoader(cat_dog_testset, batch_size=16, shuffle=False)


        # print(len(cat_dog_testset), len(cat_dog_testset))
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # todo !!
        resnet_model = CustomResnet()
        resnet_model.load_state_dict(resnet18(pretrained=False).state_dict())

        # # Parameters of newly constructed modules have requires_grad=True by default
        # for param in resnet_model.parameters():
        #     param.requires_grad = False

        # Change the prediction for 2 classes
        num_ftrs = resnet_model.fc.in_features
        resnet_model.fc = nn.Linear(num_ftrs, config.num_classes)

        resnet_model = resnet_model.to(device)

        criterion = nn.CrossEntropyLoss()

        # TODO Observe that only parameters of final layer are being optimized as opposed to before.
        optimizer_conv = optim.SGD(resnet_model.parameters(), lr=config.learning_rate_classifier1, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=6, gamma=0.1)

        try:
            checkpoint = torch.load(CHECK_POINT_PATH)
            print("checkpoint loaded")
        except:
            checkpoint = None
            print("checkpoint not found")

        _dataloaders = {'train': trainsetLoader, "val":testsetLoader}

        trainer = Trainer(_dataloaders, model=resnet_model, dataset_sizes=dataset_sizes)
        model_conv, best_val_loss, best_val_acc = trainer.train_model(criterion, optimizer_conv, exp_lr_scheduler,
                                                                      num_epochs=config.epochs,
                                                                      checkpoint=None)

        torch.save({'model_state_dict': model_conv.state_dict(),
                    'optimizer_state_dict': optimizer_conv.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_accuracy': best_val_acc,
                    'scheduler_state_dict': exp_lr_scheduler.state_dict(),
                    }, CHECK_POINT_PATH)

        print(class_names)
        print(f'Train image size: {dataset_sizes["train"]}')
        print(f'Validation image size: {dataset_sizes["val"]}')

        visualize_model(model_conv, dataloaders, class_names)

        plt.ioff()
        plt.show()


def main():
    # args = parse_args()
    wandb.login()

    config = dict(
        epochs=5,
        batch_size=32,
        learning_rate_classifier1=0.0001,
        learning_rate_classifier2=0.01,
        dataset="cats_dogs_cifar10",
        num_classes=2,
        load=True,
        CUDA=True)

    model_pipeline(config)


if __name__ == '__main__':
    main()
