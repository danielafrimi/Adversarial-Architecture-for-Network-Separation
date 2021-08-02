import torch
import wandb
import matplotlib.pyplot as plt
import torch.nn as nn


class Trainer:
    def __init__(self, trainloader, testloader, models, discrimnator):
        self.trainloader = trainloader
        self.testloader = testloader
        self.classifier1, self.classifier2 = models[0], models[1]
        self.discriminator = discrimnator
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_model(self, optimizers, optimizerD, num_epochs=20):
        if torch.cuda.device_count() > 0:
            print("gpu name", torch.cuda.get_device_name(0))

        criterionD = nn.BCELoss().to(self.device)
        criterion_classifiers = nn.CrossEntropyLoss().to(self.device)
        optimizer1, optimizer2 = optimizers[0], optimizers[1]
        opt = optimizerD
        # for visualizing the weights updates and log in wandb server
        for model, criterion in [(self.classifier1,criterion_classifiers),
                      (self.classifier2, criterion_classifiers), (self.discriminator, criterionD)]:
            wandb.watch(model, criterion, log="all", log_freq=5)


        print('Start Training')
        num_iter = 0
        self.classifier1.to(self.device)
        self.classifier2.to(self.device)
        self.discriminator.to(self.device)

        for epoch in range(num_epochs):

            running_loss_1, running_loss_2 = 0.0, 0.0

            for batch_idx, (images, targets) in enumerate(self.trainloader, 0):
                num_iter += 1



        torch.save(self.classifier1.state_dict(), './cifar_net_1.pth')
        torch.save(self.classifier2.state_dict(), './cifar_net_2.pth')
        return self.classifier1, self.classifier2

    def validation(self):
        nets_accuracy = list()
        for model in [self.classifier1, self.classifier2]:
            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data in self.testloader:
                    images, labels = data
                    images, labels = images.to(self.device), labels.to(self.device)
                    # calculate outputs by running images through the network
                    outputs, feature_map = model(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            nets_accuracy.append((100 * correct / total))
            print('Accuracy of the network on the 2000 test images: %d %%' % (100 * correct / total))

        return nets_accuracy


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
