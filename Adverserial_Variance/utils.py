import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from datasetMaker import DatasetMaker, get_class_i
from consts import LOGGER_FORMAT, DATETIME_STRING_FORMAT

def get_image_transforms():
    RC = transforms.RandomCrop(32, padding=4)
    RHF = transforms.RandomHorizontalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([TPIL,RC, RHF, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([TT, NRM])

    return transform_with_aug, transform_no_aug


def extract_subclasses_cifar10(class_1='cat', class_2='dog'):
    # Transforms object for trainset with augmentation
    transform_with_aug, transform_no_aug = get_image_transforms()

    # Downloading/Loading CIFAR10 data
    trainset = CIFAR10(root='./data', train=True, download=False)
    testset = CIFAR10(root='./data', train=False, download=False)

    classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
                 'truck': 9}

    # Separating trainset/testset data/label
    x_train = trainset.data
    x_test = testset.data
    y_train = trainset.targets
    y_test = testset.targets

    # Let's choose cats (class 3 of CIFAR) and dogs (class 5 of CIFAR) as trainset/testset
    cat_dog_trainset = DatasetMaker([get_class_i(x_train, y_train, classDict[class_1]),
                                     get_class_i(x_train, y_train, classDict[class_2])],
                                    transform_with_aug)

    cat_dog_testset = DatasetMaker([get_class_i(x_test, y_test, classDict[class_1]),
                                    get_class_i(x_test, y_test, classDict[class_2])],
                                   transform_no_aug)

    return cat_dog_trainset, cat_dog_testset

def get_trainloader_subclasses_cifar10(config):
    cat_dog_trainset, cat_dog_testset = extract_subclasses_cifar10()
    # Create datasetLoaders from trainset and testset
    trainsetLoader = DataLoader(cat_dog_trainset, batch_size=config.batch_size, shuffle=True)
    testsetLoader = DataLoader(cat_dog_testset, batch_size=config.batch_size, shuffle=False)
    return trainsetLoader, testsetLoader



def get_trainloader_all_cifar10():
    transform_with_aug, transform_no_aug = get_image_transforms()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform_with_aug)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform_no_aug)
    testloader = DataLoader(testset, batch_size=100, shuffle=False)

    return trainloader, testloader


def visualize_dataset(trainsetLoader):
    # Get a batch of training data
    inputs, classes = next(iter(trainsetLoader))

    # Make a grid from batch
    sample_train_images = torchvision.utils.make_grid(inputs)
    imshow(sample_train_images, title=None)


def imshow_img(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, dataloader, class_names, num_images=4, device='cpu'):
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])
            break


def normalize_output(self, img):
    img = img - img.min()
    img = img / img.max()
    return img

def visualize_feature_map(self, images, output, layer='conv1'):

    # Plot some images
    idx = torch.randint(0, output.size(0), ())
    img = images[idx, 0]

    fig, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img.detach().numpy())

    # Visualize feature maps
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    self.classifier1.conv1.register_forward_hook(get_activation(layer))
    data, _ = next(iter(self.testloader))
    # data.unsqueeze_(0)

    output = self.classifier1(data)

    act = activation[layer].squeeze()

    fig, axarr = plt.subplots(act.size(0))
    for idx in range(min(4, act.size(0))):
        axarr[idx].imshow(act[idx])
        plt.show()


def create_out_dir(parent_out_dir: str) -> str:
    """
    Creates the output directory in the given parent output directory,
    which will be named by the current date and time.
    """
    datetime_string = datetime.datetime.now().strftime(DATETIME_STRING_FORMAT)
    out_dir = os.path.join(parent_out_dir, datetime_string)
    os.mkdir(out_dir)

    return out_dir


def configure_logger(out_dir: str):
    """
    Configure the logger:
    (1) Remove the default logger (to stdout) and use a one with a custom format.
    (2) Adds a log file named `run.log` in the given output directory.
    """
    logger.remove()
    logger.remove()
    logger.add(sink=sys.stdout, format=LOGGER_FORMAT)
    logger.add(sink=os.path.join(out_dir, 'run.log'), format=LOGGER_FORMAT)

