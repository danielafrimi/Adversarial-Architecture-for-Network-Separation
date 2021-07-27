import torch
import wandb
import matplotlib.pyplot as plt

# Establish convention for real and fake labels during training
same_class_label = 1.
different_class_label = 0.


class Trainer:
    def __init__(self, trainloader, testloader, nets, discrimnator, path='./cifar_net.pth'):
        self.trainloader = trainloader
        self.testloader = testloader
        self.classifier1, self.classifier2 = nets[0], nets[1]
        self.weights_path = path
        self.discriminator = discrimnator
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_model(self, criterions_nets, optimizer, optimizerD, criterionD, num_epochs=10, checkpoint=None):
        criterion1, criterion2 = criterions_nets[0], criterions_nets[1]
        optimizer1, optimizer2 = optimizer[0], optimizer[1]

        # for visualizing the weights updates and log in wandb server
        wandb.watch(self.classifier1, criterion1, log="all", log_freq=5)
        wandb.watch(self.classifier2, criterion2, log="all", log_freq=5)
        wandb.watch(self.discriminator, criterion2, log="all", log_freq=5)

        print('Start Training')
        num_iter = 0
        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss_1 = 0.0
            running_loss_2 = 0.0

            train_loader_iter = iter(self.trainloader)
            for batch_idx, (images, targets) in enumerate(train_loader_iter):
                num_iter += 1

                # Split the data into 2 batches. one for each model
                images_1, images_2 = torch.split(images, 8, dim=0)
                targets_1, targets_2 = torch.split(targets, 8, dim=0)

                # Images of the same class gets the label 1, otherwise the label is 0
                same_different_class_labels = torch.logical_xor(targets_1, targets_2).type(torch.DoubleTensor)
                same_different_class_labels = same_different_class_labels < 1

                # zero the parameter gradients
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizerD.zero_grad()

                # forward + backward + optimize
                outputs1, feature_map_1 = self.classifier1(images_1)

                # forward + backward + optimize
                outputs2, feature_map_2 = self.classifier2(images_2)

                # Pass the latent code of the images to Discriminator + backward + optimize
                # same_different_class_output = self.discriminator(feature_map_1, feature_map_2)
                # loss_discriminator = criterionD(same_different_class_output, same_different_class_labels)
                # loss_discriminator.backward()
                # optimizerD.step()

                # Backward + Optimize Classifiers
                loss_1 = criterion1(outputs1, targets_1)  # todo add here
                loss_1.backward()
                optimizer1.step()

                loss_2 = criterion2(outputs2, targets_2)  #todo add here
                loss_2.backward()
                optimizer2.step()

                # print statistics
                running_loss_1 += loss_1.item()
                running_loss_2 += loss_2.item()
                # todo add d loss
                if batch_idx % 5 == 4:  # print every 2000 mini-batches
                    wandb.log({"running loss classifier_1": running_loss_1 / 200, }
                              , step=num_iter)

                    wandb.log({"running loss classifier_2": running_loss_2 / 200, }
                              , step=num_iter)

                    print('[%d, %5d] loss_1: %.3f' % (epoch + 1, batch_idx + 1, running_loss_1 / 200))
                    print('[%d, %5d] loss_2: %.3f' % (epoch + 1, batch_idx + 1, running_loss_2 / 200))
                    print()
                    running_loss_1 = 0.0
                    running_loss_2 = 0.0

                if batch_idx % 200 == 199:
                    self.validation()

        torch.save(self.classifier1.state_dict(), './cifar_net_1.pth')
        torch.save(self.classifier2.state_dict(), './cifar_net_2.pth')
        return self.classifier1, self.classifier2

    def validation(self):
        for model in [self.classifier1, self.classifier2]:
            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data in self.testloader:
                    images, labels = data
                    # calculate outputs by running images through the network
                    outputs = model(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

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
