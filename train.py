import argparse
import logging

import coloredlogs

import torch
import torch.nn as nn

import numpy as np

from sklearn.metrics import accuracy_score

import datasets.utils
import utils
from model import Predictor, ImagePredictor, Adversary

logger = logging.getLogger('Training log')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def train(dataloader_train, dataloader_val, predictor, optimizer_P, criterion, adversary, optimizer_A, scheduler, device):

    av_train_losses_P, av_train_losses_A, av_val_losses_P, av_val_losses_A = [], [], [], []
    train_accuracies_P, train_accuracies_A, val_accuracies_P, val_accuracies_A = [], [], [], []

    for epoch in range(args.n_epochs):

        # Forward (and backward when train=True) pass of the full train set
        train_losses_P, train_losses_A, labels_train_dict, protected_train_dict = utils.forward_full(dataloader_train,
                                                                                                     predictor,
                                                                                                     optimizer_P,
                                                                                                     criterion,
                                                                                                     adversary,
                                                                                                     optimizer_A,
                                                                                                     scheduler, device,
                                                                                                     train=True,
                                                                                                     images=args.image)

        # Store average training losses of predictor after every epoch
        av_train_losses_P.append(np.mean(train_losses_P))

        # Store train accuracy of predictor after every epoch
        train_accuracy_P = accuracy_score(labels_train_dict['true'], labels_train_dict['pred'])
        logger.info('Epoch {}/{}: predictor loss [train] = {:.3f}, '
                    'predictor accuracy [train] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(train_losses_P),
                                                                 train_accuracy_P))
        train_accuracies_P.append(train_accuracy_P)

        # Store train accuracy of adversary after every epoch, if applicable
        if args.debias:
            av_train_losses_A.append(np.mean(train_losses_A))
            train_accuracy_A = accuracy_score(protected_train_dict['true'], protected_train_dict['pred'])
            logger.info('Epoch {}/{}: adversary loss [train] = {:.3f}, '
                        'adversary accuracy [train] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(train_losses_A),
                                                                     train_accuracy_A))
            train_accuracies_A.append(train_accuracy_A)

        # Evaluate on validation set after every epoch, if applicable
        if args.val:
            with torch.no_grad():
                # Forward pass of full validation set
                val_losses_P, val_losses_A, labels_val_dict, protected_val_dict = \
                    utils.forward_full(dataloader_val, predictor, optimizer_P, criterion, adversary, optimizer_A,
                                       scheduler, device)

            # Store average validation losses of predictor after every epoch
            av_val_losses_P.append(np.mean(val_losses_P))

            # Store train accuracy of predictor after every epoch
            val_accuracy_P = accuracy_score(labels_val_dict['true'], labels_val_dict['pred'])
            logger.info('Epoch {}/{}: predictor loss [val] = {:.3f}, '
                        'predictor accuracy [val] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(val_losses_P),
                                                                   val_accuracy_P))
            val_accuracies_P.append(val_accuracy_P)

            if args.debias:
                av_val_losses_A.append(np.mean(val_losses_A))
                val_accuracy_A = accuracy_score(protected_val_dict['true'], protected_val_dict['pred'])
                logger.info('Epoch {}/{}: adversary loss [val] = {:.3f}, '
                            'adversary accuracy [val] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(val_losses_A),
                                                                       val_accuracy_A))
                val_accuracies_A.append(val_accuracy_A)

    logger.info('Finished training')

    # Plot accuracy and loss curves
    logger.info('Generating plots')
    utils.plot_loss_acc((av_train_losses_P, train_accuracies_P), (av_train_losses_A, train_accuracies_A))


def test(dataloader_test, predictor, optimizer_P, criterion, adversary, optimizer_A, scheduler, device):
    # Print the model parameters
    logger.info('Learned model parameters: ')
    for name, param in predictor.named_parameters():
        logger.info('Name: {}, value: {}'.format(name, param))

    # Run the model on the test set after training
    with torch.no_grad():
        test_losses_P, test_losses_A, labels_test_dict, protected_test_dict = utils.forward_full(dataloader_test,
                                                                                                 predictor, optimizer_P,
                                                                                                 criterion, adversary,
                                                                                                 optimizer_A, scheduler,
                                                                                                 device)

    test_accuracy_P = accuracy_score(labels_test_dict['true'], labels_test_dict['pred'])
    logger.info('Predictor accuracy [test] = {}'.format(test_accuracy_P))

    if args.debias:
        test_accuracy_A = accuracy_score(protected_test_dict['true'], protected_test_dict['pred'])
        logger.info('Adversary accuracy [test] = {}'.format(test_accuracy_A))

    neg_confusion_mat, neg_fpr, neg_fnr, pos_confusion_mat, pos_fpr, pos_fnr = utils.calculate_metrics(
        labels_test_dict['true'], labels_test_dict['pred'], protected_test_dict['true'])
    logger.info('Confusion matrix for the negative protected label: \n{}'.format(neg_confusion_mat))
    logger.info('FPR: {}, FNR: {}'.format(neg_fpr, neg_fnr))
    logger.info('Confusion matrix for the positive protected label: \n{}'.format(pos_confusion_mat))
    logger.info('FPR: {}, FNR: {}'.format(pos_fpr, pos_fnr))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--predictor_lr', type=float, default=0.001,
                        help='predictor learning rate')
    parser.add_argument('--adversary_lr', type=float, default=0.001,
                        help='adversary learning rate')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Frequency of evaluation on the validation set')
    parser.add_argument('--print_every', type=int, default=100,
                        help='number of iterations after which the training progress is printed')
    parser.add_argument('--debias', action='store_true',
                        help='Use the adversarial network to mitigate unwanted bias')
    parser.add_argument('--val', action="store_true",
                        help='Use a validation set during training')
    parser.add_argument('--image', action="store_true",
                        help='Use UTKFace for experiment')

    args = parser.parse_args()

    logger.info('Using configuration {}'.format(vars(args)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Using device {}'.format(device))

    # Load data
    logger.info('Loading the dataset')

    # Check whether to run experiment for UCI Adult or UTKFace dataset
    if args.image:
        dataloader_train, dataloader_val, dataloader_test = datasets.utils.get_dataloaders(args.batch_size, images=True)

        # Get feature dimension of data
        data_dims = next(iter(dataloader_train))[0].shape
        var_dims = next(iter(dataloader_train))[2].shape
        label_dims = next(iter(dataloader_train))[1].shape

        input_dim, output_dim = data_dims[1], label_dims[1]

        # Initialize the image predictor CNN

        predictor = ImagePredictor(input_dim, output_dim).to(device)
        pytorch_total_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
        logger.info(f'Number of trainable parameters: {pytorch_total_params}')

    else:
        dataloader_train, dataloader_test = datasets.utils.get_dataloaders(args.batch_size, images=False)
        dataloader_val = None

        # Get feature dimension of data
        features_dim = next(iter(dataloader_train))[0].shape[1]

        # Initialize models (for toy data the adversary is also logistic regression)
        predictor = Predictor(features_dim).to(device)

    adversary = Adversary().to(device) if args.debias else None

    # Initialize optimizers
    optimizer_P = torch.optim.Adam(predictor.parameters(), lr=args.predictor_lr)

    if args.debias:
        optimizer_A = torch.optim.Adam(adversary.parameters(), lr=args.adversary_lr)
        utils.decayer.step_count = 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_P, utils.decayer)
    else:
        optimizer_A = None
        scheduler = None

    # Setup the loss function
    criterion = nn.BCELoss()

    train(dataloader_train, dataloader_val, predictor, optimizer_P, criterion, adversary, optimizer_A, scheduler,
          device)

    test(dataloader_test, predictor, optimizer_P, criterion, adversary, optimizer_A, scheduler, device)
