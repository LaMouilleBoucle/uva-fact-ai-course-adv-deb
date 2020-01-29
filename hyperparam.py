import json
from collections import defaultdict

import argparse
import logging

import coloredlogs
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime


import datasets.utils
import utils

from sklearn.metrics import accuracy_score, mean_squared_error

from model import Predictor, ImagePredictor
from model import Adversary

logger = logging.getLogger('Training log')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def decayer(lr):
    new_lr = lr * 1000 / decayer.step_count
    decayer.step_count += 1
    return new_lr


def train(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info('Using configuration {}'.format(vars(args)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Using device {}'.format(device))

    # load data
    logger.info('Loading the dataset')
    dataloader_train, dataloader_val, dataloader_test = datasets.utils.get_dataloaders(args.batch_size, args.dataset)

    # Get feature dimension of data
    input_dim = next(iter(dataloader_train))[0].shape[1]
    protected_dim = next(iter(dataloader_train))[2].shape[1]
    output_dim = next(iter(dataloader_train))[1].shape[1]

    # Check whether to run experiment for UCI Adult or UTKFace dataset
    if args.dataset == 'images':

        # Initialize the image predictor CNN
        predictor = ImagePredictor(input_dim, output_dim).to(device)
        pytorch_total_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
        logger.info(f'Number of trainable parameters: {pytorch_total_params}')
    else:
        # Initialize models (for toy data the adversary is also logistic regression)
        predictor = Predictor(input_dim).to(device)

    adversary = Adversary(input_dim=output_dim, protected_dim=protected_dim).to(device) if args.debias else None

    # Initialize optimizers
    optimizer_P = torch.optim.Adam(predictor.parameters(), lr=args.predictor_lr)
    if args.debias:
        optimizer_A = torch.optim.Adam(adversary.parameters(), lr=args.adversary_lr)
        utils.decayer.step_count = 1
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_P, gamma=0.96)
    else:
        optimizer_A = None
        scheduler = None

    # Setup the loss function
    if args.dataset == 'crime':
        criterion = nn.MSELoss()
        metric = mean_squared_error
    else:
        criterion = nn.BCELoss()
        metric = accuracy_score

    av_train_losses_P, av_train_losses_A, av_val_losses_P, av_val_losses_A = [], [], [], []
    train_scores_P, train_scores_A, val_scores_P, val_scores_A = [], [], [], []

    for epoch in range(args.n_epochs):

        # Forward (and backward when train=True) pass of the full train set
        train_losses_P, train_losses_A, labels_train_dict, protected_train_dict = utils.forward_full(dataloader_train,
                                                                                                     predictor,
                                                                                                     optimizer_P,
                                                                                                     criterion,
                                                                                                     adversary,
                                                                                                     optimizer_A,
                                                                                                     scheduler,
                                                                                                     device,
                                                                                                     args.dataset,
                                                                                                     train=True)

        # Store average training losses of predictor after every epoch
        av_train_losses_P.append(np.mean(train_losses_P))

        # Store train accuracy of predictor after every epoch
        train_score_P = metric(labels_train_dict['true'], labels_train_dict['pred'])
        logger.info('Epoch {}/{}: predictor loss [train] = {:.3f}, '
                    'predictor score [train] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(train_losses_P),
                                                              train_score_P))
        train_scores_P.append(train_score_P)

        # Store train accuracy of adversary after every epoch, if applicable
        if args.debias:
            av_train_losses_A.append(np.mean(train_losses_A))
            train_score_A = metric(protected_train_dict['true'], protected_train_dict['pred'])
            logger.info('Epoch {}/{}: adversary loss [train] = {:.3f}, '
                        'adversary score [train] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(train_losses_A),
                                                                  train_score_A))
            train_scores_A.append(train_score_A)

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
            val_score_P = metric(labels_val_dict['true'], labels_val_dict['pred'])
            logger.info('Epoch {}/{}: predictor loss [val] = {:.3f}, '
                        'predictor score [val] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(val_losses_P),
                                                                val_score_P))

            if args.debias:
                val_score_A = metric(protected_val_dict['true'], protected_val_dict['pred'])
                logger.info('Epoch {}/{}: predictor loss [val] = {:.3f}, '
                            'predictor score [val] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(val_losses_P),
                                                                    val_score_A))

    # print parameters after training
    logger.info('Finished training')

    # Print the model parameters
    logger.info('Learned model parameters: ')
    for name, param in predictor.named_parameters():
        logger.info('Name: {}, value: {}'.format(name, param))

    # run the model on the test set after training
    with torch.no_grad():
        test_losses_P, test_losses_A, labels_test_dict, protected_test_dict, pred_y_prob, mutual_info = utils.forward_full(dataloader_test,
                                                                                                 predictor, optimizer_P,
                                                                                                 criterion, adversary,
                                                                                                 optimizer_A, scheduler,
                                                                                                 device, args.dataset)


    test_accuracy_P = metric(labels_test_dict['true'], labels_test_dict['pred'])
    logger.info('Predictor accuracy [test] = {}'.format(test_accuracy_P))

    if args.debias:
        test_accuracy_A = metric(protected_test_dict['true'], protected_test_dict['pred'])
        logger.info('Adversary accuracy [test] = {}'.format(test_accuracy_A))

    neg_confusion_mat, neg_fpr, neg_fnr, pos_confusion_mat, pos_fpr, pos_fnr = utils.calculate_metrics(
            labels_test_dict['true'], labels_test_dict['pred'], protected_test_dict['true'], args.dataset)
    logger.info('Confusion matrix for the negative protected label: \n{}'.format(neg_confusion_mat))
    logger.info('FPR: {}, FNR: {}'.format(neg_fpr, neg_fnr))
    logger.info('Confusion matrix for the positive protected label: \n{}'.format(pos_confusion_mat))
    logger.info('FPR: {}, FNR: {}'.format(pos_fpr, pos_fnr))

    return neg_confusion_mat, neg_fpr, neg_fnr, pos_confusion_mat, pos_fpr, pos_fnr, test_accuracy_P


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=30,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=20,
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
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Weight on the adversary gradient')
    parser.add_argument('--seed', type=int, default=40,
                        help='Seed used to generate other seeds for runs')
    parser.add_argument('--dataset', type=str, default='adult',
                        help='Tabular dataset to be used: adult, crime, images')

    args = parser.parse_args()

    lr_P = [0.001, 0.01, 0.1]
    lr_A = [0.001, 0.01, 0.1]
    batch = 128
    alphas = np.linspace(start=0.1, stop=1.0, num=10)

    data = defaultdict(lambda: defaultdict(list))

    file_name = "data-" + str(datetime.now()).replace(':', '-').replace(' ', '_') + ".json"

    if args.debias:
        file_name = "data_debias-" + str(datetime.now()).replace(':', '-').replace(' ', '_') + ".json"

    for alpha in alphas:
        for p in lr_P:
            for a in lr_A:

                key = str((p, a, batch, alpha))

                for i in range(5):
                    args.predictor_lr = p
                    args.adversary_lr = a
                    args.batch_size = batch
                    args.alpha = alpha

                    neg_confusion_mat, neg_fpr, neg_fnr, pos_confusion_mat, pos_fpr, pos_fnr, predictor_acc = train(args.seed+i)

                    data[key]["neg_fpr"].append(neg_fpr)
                    data[key]["neg_fnr"].append(neg_fnr)
                    data[key]["pos_fpr"].append(pos_fpr)
                    data[key]["pos_fnr"].append(pos_fnr)
                    data[key]["predictor_acc"].append(predictor_acc)


                    data[key]["neg_confusion_mat"].append(neg_confusion_mat.tolist())
                    data[key]["pos_confusion_mat"].append(pos_confusion_mat.tolist())

                with open(file_name, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

    logger.info('Results written to file: {}'.format(file_name))
