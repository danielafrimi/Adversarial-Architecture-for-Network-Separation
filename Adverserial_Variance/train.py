import torch
import wandb
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from loguru import logger
from consts import CLASSES_BINARY

different_class_label = 0.


class Trainer:
    def __init__(self, trainloader, testloader, models, discriminator, cross_entropy_loss_weight):
        self.trainloader = trainloader
        self.testloader = testloader
        self.classifier1, self.classifier2 = models[0], models[1]
        self.discriminator = discriminator
        self.cross_entropy_loss_weight = cross_entropy_loss_weight
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_model(self, optimizers, optimizerD, use_discriminator, num_epochs=20):

        criterionD = nn.BCELoss().to(self.device)
        criterion_classifiers = nn.CrossEntropyLoss().to(self.device)
        optimizer1, optimizer2 = optimizers[0], optimizers[1]

        # for visualizing the weights updates and log in wandb server
        for model, criterion in [(self.classifier1, criterion_classifiers),
                                 (self.classifier2, criterion_classifiers), (self.discriminator, criterionD)]:
            wandb.watch(model, criterion, log="all", log_freq=5)

        print('Start Training')
        num_iter = 0

        # Run models to GPU
        self.classifier1.to(self.device)
        self.classifier2.to(self.device)

        if use_discriminator:
            self.discriminator.to(self.device)

        for epoch in range(num_epochs):

            running_loss_1, running_loss_2, running_loss_d = 0.0, 0.0, 0.0

            for batch_idx, (images, targets) in enumerate(self.trainloader, 0):
                num_iter += 1

                # Split the data into 2 batches. one batch for each model
                images_1, images_2 = torch.split(images, images.shape[0] // 2, dim=0)
                targets_1, targets_2 = torch.split(targets, targets.shape[0] // 2, dim=0)

                # Move to GPU
                images_1, images_2, targets_1, targets_2 = images_1.to(self.device), images_2.to(self.device), \
                                                           targets_1.to(self.device), targets_2.to(self.device)

                discriminator_labels = self.get_discriminator_labels(targets_1, targets_2)

                # zero the parameter gradients
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizerD.zero_grad()

                # forward + backward + optimize
                # todo feature map is a tensor of shape torch.Size([32, 4096]), 32 is the batch size
                outputs_1, feature_map = self.classifier1(images_1)
                outputs_2, feature_map2 = self.classifier2(images_2)
                if use_discriminator:
                    discriminator_output = self.discriminator(feature_map, feature_map2)
                    loss_discriminator = criterionD(discriminator_output, discriminator_labels)

                    loss_discriminator.backward(retain_graph=True)

                # The labels of the discriminator
                batch_size = images_1.size(0)
                label_classifier_D = torch.full((batch_size,), different_class_label, dtype=torch.float,
                                                device=self.device)

                # a = self.cross_entropy_loss_weight * criterion(outputs_1, targets_1)
                # b = (1.0 - self.cross_entropy_loss_weight) * criterionD(discriminator_output, label_classifier_D)
                if use_discriminator:
                    loss_1 = self.cross_entropy_loss_weight * criterion_classifiers(outputs_1, targets_1) + \
                             (1.0 - self.cross_entropy_loss_weight) * criterionD(discriminator_output,
                                                                                 label_classifier_D)

                    loss_2 = self.cross_entropy_loss_weight * criterion_classifiers(outputs_2, targets_2) + \
                             (1.0 - self.cross_entropy_loss_weight) * criterionD(discriminator_output,
                                                                                 label_classifier_D)
                else:
                    loss_1 = criterion_classifiers(outputs_1, targets_1)
                    loss_2 = criterion_classifiers(outputs_2, targets_2)

                loss_1.backward(retain_graph=True)
                loss_2.backward(retain_graph=True)

                if use_discriminator:
                    optimizerD.step()
                optimizer1.step()
                optimizer2.step()

                # print statistics
                running_loss_1 += loss_1.item()
                running_loss_2 += loss_2.item()
                if use_discriminator:
                    running_loss_d += loss_discriminator.item()

                if batch_idx % 20 == 19:
                    # Log statics
                    for loss, model_name in [(running_loss_1, 'classifier_1'), (running_loss_2, 'classifier_2'),
                                             (running_loss_d, 'disc')]:
                        wandb.log({"running loss {}".format(model_name): loss / 20, }, step=num_iter)
                        logger.info({"running loss {}".format(model_name): loss / 20})

                    running_loss_1, running_loss_2, running_loss_d = 0.0, 0.0, 0.0

            self.validation(num_iter)

        print('Finished Training')
        self.save_models()

    def get_discriminator_labels(self, targets_1, targets_2):
        # Images of the same class gets the label 1, otherwise the label is 0
        same_different_class_labels_numpy = np.logical_xor(targets_1.cpu().numpy(),
                                                           targets_2.cpu().numpy()).astype(int)
        same_different_class_labels_numpy = (same_different_class_labels_numpy < 1)

        discriminator_labels = torch.from_numpy(same_different_class_labels_numpy)
        discriminator_labels = discriminator_labels.type(torch.FloatTensor)
        discriminator_labels = torch.unsqueeze(discriminator_labels, dim=1).to(self.device)

        return discriminator_labels

    def ensemble_models(self):
        pass

    def validation(self, num_iter):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for id, model in enumerate([self.classifier1, self.classifier2]):
                for data in self.testloader:
                    images, labels = data[0].to(self.device), data[1].to(self.device)
                    # calculate outputs by running images through the network
                    outputs, f = model(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                print('Accuracy of the network on test images: %d %%' % (100 * correct / total))
                wandb.log({"Accuracy Model {}".format(id + 1): (100 * correct / total), }
                          , step=num_iter)

    def save_models(self):
        torch.save(self.classifier1.state_dict(), './cifar_net_1.pth')
        torch.save(self.classifier2.state_dict(), './cifar_net_2.pth')
        torch.save(self.discriminator.state_dict(), './cifar_net_d.pth')
