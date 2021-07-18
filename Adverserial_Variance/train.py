import torch
import wandb
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, trainloader, testloader, nets, discrimnator, path='./cifar_net.pth'):
        self.trainloader = trainloader
        self.testloader = testloader
        self.classifier1, self.classifier2 = nets[0], nets[1]
        self.weights_path = path
        self.discrimnator = discrimnator
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_model(self, criterion, optimizer, num_epochs=10, checkpoint=None):
        # for visualizing the weights updates and log in wandb server
        criterion1, criterion2 = criterion[0], criterion[1]
        optimizer1, optimizer2 = optimizer[0], optimizer[1]

        wandb.watch(self.classifier1, criterion1, log="all", log_freq=5)
        wandb.watch(self.classifier2, criterion2, log="all", log_freq=5)

        num_iter = 0
        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss_1 = 0.0
            running_loss_2 = 0.0

            train_loader_iter = iter(self.trainloader)
            for batch_idx, (images, targets) in enumerate(train_loader_iter):
                num_iter += 1

                # fetch second batch
                images_2, targets_2 = next(train_loader_iter)
                batch_idx += 1

                # zero the parameter gradients
                optimizer1.zero_grad()
                optimizer2.zero_grad()

                # forward + backward + optimize
                outputs1 = self.classifier1(images)
                loss_1 = criterion1(outputs1, targets)
                loss_1.backward()
                optimizer1.step()

                # forward + backward + optimize
                outputs2 = self.classifier2(images_2)
                loss_2 = criterion2(outputs2, targets_2)
                loss_2.backward()
                optimizer2.step()

                # print statistics
                running_loss_1 += loss_1.item()
                running_loss_2 += loss_2.item()
                if batch_idx % 200 == 199:  # print every 2000 mini-batches
                    wandb.log({"running loss classifier_1": running_loss_1 / 200, }
                              , step=num_iter)

                    wandb.log({"running loss classifier_2": running_loss_2 / 200, }
                              , step=num_iter)

                if batch_idx % 2000 == 1999:
                    self.visualize_feature_map(images, outputs1)

                    print('[%d, %5d] loss_1: %.3f' % (epoch + 1, batch_idx + 1, running_loss_1 / 200))
                    print('[%d, %5d] loss_2: %.3f' % (epoch + 1, batch_idx + 1, running_loss_2 / 200))
                    print()
                    running_loss_1 = 0.0
                    running_loss_2 = 0.0

        torch.save(self.classifier1.state_dict(), self.weights_path)
        torch.save(self.classifier2.state_dict(), self.weights_path)
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

    def visualize_feature_map(self, images, output):

        # Plot some images
        idx = torch.randint(0, output.size(0), ())
        # pred = self.normalize_output(output[idx, 0])
        img = images[idx, 0]

        fig, axarr = plt.subplots(1, 2)
        axarr[0].imshow(img.detach().numpy())

        # Visualize feature maps
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        self.classifier1.conv1.register_forward_hook(get_activation('conv1'))
        data, _ = next(iter(self.testloader))
        # data.unsqueeze_(0)

        output = self.classifier1(data)

        act = activation['conv1'].squeeze()

        fig, axarr = plt.subplots(act.size(0))
        for idx in range(act.size(0)):
            axarr[idx].imshow(act[idx])
        plt.show()
