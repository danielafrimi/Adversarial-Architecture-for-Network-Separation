import torch
import wandb


class Trainer:
    def __init__(self, trainloader, testloader, nets, path='./cifar_net.pth'):
        self.trainloader = trainloader
        self.testloader = testloader
        self.classifier1, self.classifier2 = nets[0], nets[1]
        self.weights_path = path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_model(self, criterion, optimizer, num_epochs=10, checkpoint=None):
        # for visualizing the weights updates and log in wandb server
        criterion1, criterion2 = criterion[0], criterion[1]
        optimizer1, optimizer2 = optimizer[0], optimizer[1]

        wandb.watch(self.classifier1, criterion1, log="all", log_freq=5)
        wandb.watch(self.classifier2, criterion2, log="all", log_freq=5)

        num_iter = 0
        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                num_iter += 1
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 200 == 199:  # print every 2000 mini-batches
                    wandb.log({"running loss": running_loss / 200, }
                              , step=num_iter)

                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        torch.save(self.net.state_dict(), self.weights_path)
        return self.net

    def validation(self):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self.net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
