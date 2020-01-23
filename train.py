import argparse
import logging
import coloredlogs

import torch
import torch.nn as nn

import numpy as np

from sklearn.metrics import accuracy_score

import utils
from model import Predictor, ImagePredictor, Adversary

logger = logging.getLogger('Training log')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def train():
    global logger

    logger.info('Using configuration {}'.format(vars(args)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Using device {}'.format(device))

    # load data
    logger.info('Loading the dataset')

    # check whether to run experiment for UCI Adult or UTKFace datasets
    if args.image:
        dataloader_train, dataloader_val, dataloader_test = utils.get_dataloaders(args.batch_size, logger, images=True)

        # get feature dimension of data
        data_dims = next(iter(dataloader_train))[0].shape
        var_dims = next(iter(dataloader_train))[2].shape
        label_dims = next(iter(dataloader_train))[1].shape
        
        input_dim, output_dim = data_dims[1], label_dims[1]

        # Initialize the image predictor CNN

        predictor = ImagePredictor(input_dim, output_dim).to(device)
        pytorch_total_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
        logger.info(f'Number of trainable parameters: {pytorch_total_params}')

    else:
        dataloader_train, dataloader_test = utils.get_dataloaders(args.batch_size, logger, images=False)

        # get feature dimension of data
        features_dim = next(iter(dataloader_train))[0].shape[1]
        # Initialize models (for toy data the adversary is also logistic regression)
        predictor = Predictor(features_dim).to(device)

    adversary =  Adversary().to(device) if args.debias else None

    # initialize optimizers
    optimizer_P = torch.optim.Adam(predictor.parameters(), lr=args.predictor_lr)

    if args.debias:
        optimizer_A = torch.optim.Adam(adversary.parameters(), lr=args.adversary_lr)

        utils.decayer.step_count = 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_P, utils.decayer)
    else:
        optimizer_A = None
        scheduler = None

    # setup the loss function
    criterion = nn.BCELoss()

    model = (predictor, adversary, criterion, optimizer_P, optimizer_A, scheduler, device)

    av_train_losses_P = []
    av_train_losses_A = []
    av_val_losses_P = []
    av_val_losses_A = []

    train_accuracies_P = []
    train_accuracies_A = []
    val_accuracies_P = []
    val_accuracies_A = []

    for epoch in range(args.n_epochs):

        train_losses_P = []
        train_losses_A = []
        val_losses_P = []
        val_losses_A = []

        labels_train = {'true': [], 'pred': []}
        labels_val = {'true': [], 'pred': []}

        protected_train = {'true': [], 'pred': []}
        protected_val = {'true': [], 'pred': []}

        train_predictions_P = []
        train_predictions_A = []
        val_predictions_P = []
        val_predictions_A = []

        train_logger = (labels_train, protected_train,\
                        (train_losses_P, train_losses_A),\
                        (train_predictions_P, train_predictions_A))

        val_logger = (labels_val, protected_val,\
                        (val_losses_P, val_losses_A),\
                        (val_predictions_P, val_predictions_A))

        # Forward (and backward, because train=True) pass of the full train set
        utils.forward_full(dataloader_train, model, train_logger, train=True, images=args.image)

        # store average training losses of predictor after every epoch
        av_train_losses_P.append(np.mean(train_losses_P))

        # store train accuracy of predictor after every epoch
        # print(labels_train['true'])
        # print(labels_train['pred'])
        train_accuracy_P = accuracy_score(labels_train['true'], labels_train['pred'])
        logger.info('Epoch {}/{}: predictor loss [train] = {:.3f}, '
                    'predictor accuracy [train] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(train_losses_P), train_accuracy_P))
        train_accuracies_P.append(train_accuracy_P)

        # store train accuracy of adversary after every epoch, if applicable
        if args.debias:
            av_train_losses_A.append(np.mean(train_losses_A))
            train_accuracy_A = accuracy_score(protected_train['true'], protected_train['pred'])
            logger.info('Epoch {}/{}: adversary loss [train] = {:.3f}, '
                        'adversary accuracy [train] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(train_losses_A),
                                                                     train_accuracy_A))
            train_accuracies_A.append(train_accuracy_A)

        # evaluate on validation set after every epoch, if applicable
        if args.val:
            with torch.no_grad():
                # Forward pass of full validation set
                utils.forward_full(dataloader_val, model, val_logger)

            # store average validation losses of predictor after every epoch
            av_val_losses_P.append(np.mean(val_losses_P))

            # store train accuracy of predictor after every epoch
            val_accuracy_P = accuracy_score(labels_val['true'], labels_val['pred'])
            logger.info('Epoch {}/{}: predictor loss [val] = {:.3f}, '
                        'predictor accuracy [val] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(val_losses_P), val_accuracy_P))
            val_accuracies_P.append(val_accuracy_P)

            if args.debias:
                av_val_losses_A.append(np.mean(val_losses_A))
                # val_accuracy_A = accuracy_score(val_dataset.z.squeeze(dim=1).numpy(), val_predictions_A)
                val_accuracy_A = accuracy_score(protected_val['true'], protected_val['pred'])
                logger.info('Epoch {}/{}: adversary loss [val] = {:.3f}, '
                            'adversary accuracy [val] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(val_losses_A),
                                                                       val_accuracy_A))
                val_accuracies_A.append(val_accuracy_A)

    # print parameters after training
    logger.info('Finished training')

    # Print the model parameters
    logger.info('Learned model parameters: ')
    for name, param in predictor.named_parameters():
        logger.info('Name: {}, value: {}'.format(name, param))

    test_predictions_P = []
    test_predictions_A = []
    labels_test = {'true': [], 'pred': []}
    protected_test = {'true': [], 'pred': []}

    test_logger = (labels_test, protected_test,\
                   ([], []),\
                   (test_predictions_P, test_predictions_A))

    # run the model on the test set after training
    with torch.no_grad():
        utils.forward_full(dataloader_test, model, test_logger)

    test_accuracy_P = accuracy_score(labels_test['true'], labels_test['pred'])
    logger.info('Predictor accuracy [test] = {}'.format(test_accuracy_P))

    if args.debias:
        test_accuracy_A = accuracy_score(protected_test['true'], protected_test['pred'])
        logger.info('Adversary accuracy [test] = {}'.format(test_accuracy_A))

    neg_confusion_mat, neg_fpr, neg_fnr, pos_confusion_mat, pos_fpr, pos_fnr = utils.calculate_metrics(labels_test['true'], labels_test['pred'], protected_test['true'])
    logger.info('Confusion matrix for the negative protected label: \n{}'.format(neg_confusion_mat))
    logger.info('FPR: {}, FNR: {}'.format(neg_fpr, neg_fnr))
    logger.info('Confusion matrix for the positive protected label: \n{}'.format(pos_confusion_mat))
    logger.info('FPR: {}, FNR: {}'.format(pos_fpr, pos_fnr))

    # plot accuracy and loss curves
    logger.info('Generating plots')
    utils.plot_loss_acc((av_train_losses_P,train_accuracies_P), (av_train_losses_A,train_accuracies_A))


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
    parser.add_argument('--debias',  action='store_true',
                        help='Use the adversial network to mitigate unwanted bias')
    parser.add_argument('--val',  action="store_true",
                        help='Use a validation set during training')
    parser.add_argument('--image',  action="store_true",
                        help='Use UTKFace for experiment')


    args = parser.parse_args()

    train()
