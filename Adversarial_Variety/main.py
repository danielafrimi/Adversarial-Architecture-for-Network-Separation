import argparse

import torch.optim as optim
import wandb
from loguru import logger

from models.simple_resnet import Simple_Resnet
from models.disciminator import CustomDiscriminator, SiameseDiscriminator
from models.resnet import ResNet18
from train import Trainer
from utils import get_trainloader_subclasses_cifar10


def parse_args():
    parser = argparse.ArgumentParser(
        description='Main script. '
                    'This enables running the different experiments while logging to a log-file and to wandb.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--experiment_type", type=str, default='only_classifiers', choices=['all', 'only_classifiers'],
                        help=f'')

    # Arguments defining the model.
    parser.add_argument('--num_models', type=int, default=2, help=f'The number of classifier to train in one train')
    parser.add_argument('--model', type=str, default='resnet',
                        choices=['basic_cnn', 'resnet'], help=f'The model name for the classifier network architecture')
    parser.add_argument('--model_discriminator', type=str, default='CustomDiscriminator',
                        choices=['CustomDiscriminator', 'SiameseDiscriminator'], help=f'The model name for the '
                                                                                      f'Discriminator network '
                                                                                      f'architecture')

    # Arguments defining the training-process
    parser.add_argument('--batch_size', type=int, default=256, help=f'Batch size')
    parser.add_argument('--epochs', type=int, default=300, help=f'Number of epochs')
    parser.add_argument('--learning_rate_classifiers', type=float, default=0.001, help=f'Learning-rate of classifier')
    parser.add_argument('--learning_rate_discriminator', type=float, default=0.003,
                        help=f'Learning-rate of discriminator')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help=f'Weight decay_classifiers')
    parser.add_argument('--cross_entropy_loss_weight', type=float, default=0.6,
                        help=f'scalar (between 0-1), that indicates how much weight to give to cross_entropy_loss '
                             f'(discrimniator loss weight is 1 - cross_entropy_loss_weight')

    # Arguments for logging the training process.
    parser.add_argument('--path', type=str, default='./experiments', help=f'Output path for the experiment - '
                                                                          f'a sub-directory named with the data and '
                                                                          f'time will be created within')
    parser.add_argument('--log_interval', type=int, default=100,
                        help=f'How many iterations between each training log')

    return parser.parse_args()


def get_classifier_model(args):
    models = dict()

    for i in range(args.num_models):
        model = ResNet18() if args.model == 'resnet' else Simple_Resnet(num_classes=2)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate_classifiers,
                              momentum=0.9, weight_decay=args.weight_decay)

        models[model] = optimizer

    return models


def get_discriminator(args):
    global discriminator
    if args.model_discriminator == 'CustomDiscriminator':
        discriminator = CustomDiscriminator()
    elif args.model_discriminator == 'SiameseDiscriminator':
        discriminator = SiameseDiscriminator()

    optimizerD = optim.Adam(discriminator.parameters(), lr=args.learning_rate_discriminator, betas=(0.5, 0.999))
    return discriminator, optimizerD


def model_pipeline(hyperparameters):
    # Tell wandb to get started
    with wandb.init(project="pytorch-demo", config=hyperparameters):
        # access all HPs through wandb.config, so logging matches execution!
        args = wandb.config

        logger.info(f'Starting to train {args.model} '
                    f'Experiment type is {args.experiment_type} '
                    f'for {args.epochs} epochs '
                    f'cross_entropy_loss_weight is {args.cross_entropy_loss_weight} '
                    f'bs={args.batch_size}, '
                    f'lr_classifier={args.learning_rate_classifiers}, '
                    f'lr_D={args.learning_rate_discriminator}, '
                    f'wd={args.weight_decay}')

        trainsetLoader, testsetLoader = get_trainloader_subclasses_cifar10(args)

        models = get_classifier_model(args)

        discriminator, optimizerD = get_discriminator(args)

        trainer = Trainer(trainloader=trainsetLoader, testloader=testsetLoader,
                          models=list(models.keys()), discriminator=discriminator,
                          cross_entropy_loss_weight=args.cross_entropy_loss_weight)
        # Train
        trainer.train_model(optimizers=list(models.values()), optimizerD=optimizerD, num_epochs=args.epochs,
                            use_discriminator=args.experiment_type == 'all')


@logger.catch
def main():
    args = parse_args()

    model_pipeline(hyperparameters=args)


if __name__ == '__main__':
    main()
