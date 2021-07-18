import copy
import math
import time

import torch
import wandb


class Trainer:
    def __init__(self, dataloaders, model, dataset_sizes):
        self.dataloaders = dataloaders
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset_sizes = dataset_sizes

    def train_model(self, criterion, optimizer, scheduler, num_epochs=10, checkpoint=None):
        # for visualizing the weights updates and log in wandb server
        wandb.watch(self.model, criterion, log="all", log_freq=5)

        since = time.time()

        if checkpoint is None:
            best_model_wts = copy.deepcopy(self.model.state_dict())
            best_loss = math.inf
            best_acc = 0.
        else:
            print(f'Val loss: {checkpoint["best_val_loss"]}, Val accuracy: {checkpoint["best_val_accuracy"]}')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            best_model_wts = copy.deepcopy(self.model.state_dict())
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_loss = checkpoint['best_val_loss']
            best_acc = checkpoint['best_val_accuracy']

        iter_number = 0
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for i, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    iter_number += 1
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    if i % 10 == 9:
                        wandb.log({"running loss": running_loss / (i * inputs.size(0)), }
                                  , step=iter_number)

                        print('[%d, %d] loss: %.3f' % (epoch + 1, i, running_loss / (i * inputs.size(0))))

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                wandb.log({"Loss:": epoch_loss, "Acc": epoch_acc, }, step=iter_number)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    print(f'New best model found!')
                    print(f'New record loss: {epoch_loss}, previous record loss: {best_loss}')
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:.4f} Best val loss: {:.4f}'.format(best_acc, best_loss))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model, best_loss, best_acc