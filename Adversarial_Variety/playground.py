import torch
import torchvision
import torchvision.transforms as transforms
from models.resnet import ResNet18
import wandb
from loguru import logger
from models.disciminator import CustomDiscriminator

from torch.utils.data import DataLoader


from models.simple_resnet import Simple_Resnet

print(torch.__version__)

same_class_label = 1.
different_class_label = 0.

hyperparameters = dict(
    epochs=80,
    batch_size=256,
    cross_entropy_loss_weight=0.8,
    disc_loss_weight=0.2,
    learning_rate_classifier1=0.001,
    learning_rate_classifier2=0.001,
    dataset="cifar10",
    num_classes=2)


@logger.catch
def main():
    with wandb.init(project="pytorch-demo", config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        logger.add("resnet_log.log", rotation="1 week")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(device)


        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Normalize the test set same as training set without augmentation
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=False, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size,
                                                 shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        classes_binary = ('cat', 'dog')

        cat_dog_trainset, cat_dog_testset = extract_specific_classes_cifar10()

        # Create datasetLoaders from trainset and testset
        trainsetLoader = DataLoader(cat_dog_trainset, batch_size=config.batch_size, shuffle=True)
        testsetLoader = DataLoader(cat_dog_testset, batch_size=4, shuffle=True)

        import matplotlib.pyplot as plt
        import numpy as np

        # functions to show an image

        def imshow(img):
            img = img / 2 + 0.5  # unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()

        # # get some random training images
        # dataiter = iter(trainsetLoader)
        # images, labels = dataiter.next()
        #
        # # show images
        # imshow(torchvision.utils.make_grid(images))
        # # print labels
        # print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

        import torch.nn as nn
        import torch.nn.functional as F

        net = Simple_Resnet()
        net2 = Simple_Resnet()
        discriminator = CustomDiscriminator()

        import torch.optim as optim

        criterion = nn.CrossEntropyLoss()
        criterion2 = nn.CrossEntropyLoss()
        criterionD = nn.BCELoss()
        optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer = optim.SGD(net.parameters(), lr=config.learning_rate_classifier1, momentum=0.9)
        optimizer2 = optim.SGD(net2.parameters(), lr=config.learning_rate_classifier1, momentum=0.9)

        wandb.watch(net, criterion, log="all", log_freq=5)
        wandb.watch(net2, criterion2, log="all", log_freq=5)
        wandb.watch(discriminator, criterionD, log="all", log_freq=5)

        num_iter = 0

        net.to(device)
        net2.to(device)
        discriminator.to(device)

        for epoch in range(config.epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            running_loss_2 = 0.0
            running_loss_d = 0.0
            for i, data in enumerate(trainsetLoader, 0):
                num_iter += 1

                # Split the data into 2 batches. one batch for each model
                images_1, images_2 = torch.split(data[0], data[0].shape[0] // 2, dim=0)
                targets_1, targets_2 = torch.split(data[1], data[1].shape[0] // 2, dim=0)
                # Move to GPU
                images_1, images_2, targets_1, targets_2 = images_1.to(device), images_2.to(device), \
                                                           targets_1.to(device), targets_2.to(device)

                # Images of the same class gets the label 1, otherwise the label is 0
                same_different_class_labels_numpy = np.logical_xor(targets_1.cpu().numpy(),
                                                                   targets_2.cpu().numpy()).astype(int)
                labels_classifiers = torch.from_numpy(same_different_class_labels_numpy).type(torch.FloatTensor)
                labels_classifiers = torch.unsqueeze(labels_classifiers, dim=1).to(device)
                same_different_class_labels_numpy = (same_different_class_labels_numpy < 1)
                discriminator_labels = torch.from_numpy(same_different_class_labels_numpy)
                discriminator_labels = discriminator_labels.type(torch.FloatTensor)
                discriminator_labels = torch.unsqueeze(discriminator_labels, dim=1).to(
                    device)

                # zero the parameter gradients
                optimizer.zero_grad()
                optimizer2.zero_grad()
                optimizerD.zero_grad()

                # forward + backward + optimize
                # todo feature map is a tensor of shape torch.Size([32, 4096]), 32 is the batch size
                outputs, feature_map = net(images_1)
                outputs2, feature_map2 = net2(images_2)

                discriminator_output = discriminator(feature_map, feature_map2)
                loss_discriminator = criterionD(discriminator_output,
                                                discriminator_labels)

                loss_discriminator.backward(retain_graph=True)

                # todo create tensors of 1 ?
                b_size = images_1.size(0)
                label_classifier_D = torch.full((b_size,), different_class_label, dtype=torch.float, device=device)
                a = config.cross_entropy_loss_weight * criterion(outputs, targets_1)
                b = config.disc_loss_weight * criterionD(discriminator_output, label_classifier_D)
                loss = a + b

                loss2 = config.cross_entropy_loss_weight * criterion2(outputs2, targets_2) + \
                        config.disc_loss_weight * criterionD(discriminator_output, label_classifier_D)

                loss.backward(retain_graph=True)
                loss2.backward(retain_graph=True)

                optimizerD.step()
                optimizer.step()
                optimizer2.step()

                # print statistics
                running_loss += loss.item()
                running_loss_2 += loss2.item()
                running_loss_d += loss_discriminator.item()

                if i % 20 == 19:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 70))
                    wandb.log({"running loss classifier_1": running_loss / 70, }
                              , step=num_iter)
                    wandb.log({"running loss classifier_2": running_loss_2 / 70, }
                              , step=num_iter)
                    wandb.log({"running loss d": running_loss_d / 70, }
                              , step=num_iter)
                    logger.info({"running loss classifier_1": running_loss / 70})
                    logger.info({"running loss classifier_2": running_loss_2 / 70})
                    logger.info({"running loss D": running_loss_d / 70})

                    running_loss = 0.0
                    running_loss_2 = 0.0
                    running_loss_d = 0.0

            validation(net, net2, testsetLoader, device, classes_binary, classes, num_iter)

        print('Finished Training')


def validation(net, net2, testsetLoader, device, classes_binary, classes, num_iter):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for id, model in enumerate([net, net2]):
            for data in testsetLoader:
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs, f = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the network on test images: %d %%' % (100 * correct / total))
            wandb.log({"Accuracy Model {}".format(id + 1): (100 * correct / total), }
                      , step=num_iter)

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes_binary}
    total_pred = {classname: 0 for classname in classes}

    # visualize_model(net, testsetLoader, classes_binary, device=device)

    # # again no gradients needed
    # with torch.no_grad():
    #     for model in [net, net2]:
    #         for data in testsetLoader:
    #             images, labels = data[0].to(device), data[1].to(device)
    #             # cat label is 0 and dog is 1
    #             outputs, f = model(images)
    #             # imshow_img(torchvision.utils.make_grid(images.cpu()))
    #             _, predictions = torch.max(outputs, 1)
    #             # collect the correct predictions for each class
    #             for label, prediction in zip(labels, predictions):
    #                 if label == prediction:
    #                     correct_pred[classes_binary[label]] += 1
    #                 total_pred[classes_binary[label]] += 1
    #
    #         print(correct_pred.items())
    #         # print accuracy for each class
    #         for classname, correct_count in correct_pred.items():
    #             accuracy = 100 * float(correct_count) / total_pred[classname]
    #             print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
    #                                                                  accuracy))


if __name__ == '__main__':
    main()
