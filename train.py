import argparse
import logging
import os

import coloredlogs

import torch
import torch.nn as nn

import numpy as np

from sklearn.metrics import accuracy_score, mean_squared_error

import datasets.utils
import utils
from model import Predictor, ImagePredictor, Adversary

logger = logging.getLogger('Training log')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def train(dataloader_train, dataloader_val, predictor, optimizer_P, criterion, metric, adversary, optimizer_A, scheduler, device):

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

        # Store train performance of predictor after every epoch
        train_score_P = metric(labels_train_dict['true'], labels_train_dict['pred'])
        logger.info('Epoch {}/{}: predictor loss [train] = {:.3f}, '
                    'predictor score [train] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(train_losses_P),
                                                              train_score_P))
        train_scores_P.append(train_score_P)

        # Store train performance of adversary after every epoch, if applicable
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
                val_losses_P, val_losses_A, labels_val_dict, protected_val_dict, _ = \
                    utils.forward_full(dataloader_val, predictor, optimizer_P, criterion, adversary, optimizer_A,
                                       scheduler, device, args.dataset)

            # Store average validation losses of predictor after every epoch
            av_val_losses_P.append(np.mean(val_losses_P))

            # Store train accuracy of predictor after every epoch
            val_score_P = metric(labels_val_dict['true'], labels_val_dict['pred'])
            logger.info('Epoch {}/{}: predictor loss [val] = {:.3f}, '
                        'predictor score [val] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(val_losses_P),
                                                                val_score_P))

            if args.debias:
                val_score_A = metric(protected_val_dict['true'], protected_val_dict['pred'])
                logger.info('Epoch {}/{}: adversary loss [val] = {:.3f}, '
                            'adversary score [val] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(val_losses_A),
                                                                    val_score_A))

    logger.info('Finished training')

    # Plot learning curves
    logger.info('Generating plots')
    utils.plot_loss_acc((av_train_losses_P, train_scores_P, av_val_losses_P, val_scores_P),
                        (av_train_losses_A, train_scores_A, av_val_losses_A, val_scores_A),
                        args.dataset)

    if args.dataset == 'crime':
        utils.make_coplot(protected_val_dict, labels_val_dict)

    # Save model
    os.makedirs(args.save_model_to, exist_ok=True)
    torch.save(predictor.state_dict(), args.save_model_to + "pred_biased_"+str(args.debias)+"_"+str(args.dataset)+"_seed_"+str(args.seed))
    if args.debias:
        torch.save(predictor.state_dict(), args.save_model_to + "adv_seed_"+str(args.seed))



def test(dataloader_test, predictor, optimizer_P, criterion, metric, adversary, optimizer_A, scheduler, device):
    # Print the model parameters
    logger.info('Learned model parameters: ')
    for name, param in predictor.named_parameters():
        logger.info('Name: {}, value: {}'.format(name, param))

    # Run the model on the test set after training
    with torch.no_grad():
        test_losses_P, test_losses_A, labels_test_dict, protected_test_dict, pred_y_prob = utils.forward_full(dataloader_test,
                                                                                                 predictor, optimizer_P,
                                                                                                 criterion, adversary,
                                                                                                 optimizer_A, scheduler,
                                                                                                 device, args.dataset)

    test_score_P = metric(labels_test_dict['true'], labels_test_dict['pred'])
    logger.info('Predictor score [test] = {}'.format(test_score_P))

    if args.debias:
        test_score_A = metric(protected_test_dict['true'], protected_test_dict['pred'])
        logger.info('Adversary score [test] = {}'.format(test_score_A))

    if args.dataset == 'adult':
        neg_confusion_mat, neg_fpr, neg_fnr, pos_confusion_mat, pos_fpr, pos_fnr = utils.calculate_metrics(
            labels_test_dict['true'], labels_test_dict['pred'], protected_test_dict['true'], args.dataset)
        logger.info('Confusion matrix for the negative protected label: \n{}'.format(neg_confusion_mat))
        logger.info('FPR: {}, FNR: {}'.format(neg_fpr, neg_fnr))
        logger.info('Confusion matrix for the positive protected label: \n{}'.format(pos_confusion_mat))
        logger.info('FPR: {}, FNR: {}'.format(pos_fpr, pos_fnr))
    elif args.dataset == 'images':
        m_prec, m_recall, m_fscore, m_support, m_auc, w_prec, w_recall, w_fscore, w_support, w_auc, avg_dif, avg_abs_dif = utils.calculate_metrics(
            labels_test_dict['true'], labels_test_dict['pred'], protected_test_dict['true'], args.dataset, pred_probs=pred_y_prob)
        logger.info(f'For men: precision {m_prec}, recall {m_recall}, F1 {m_fscore}, support {m_support}, AUC {m_auc}.')
        logger.info(f'For women: precision {w_prec}, recall {w_recall}, F1 {w_fscore}, support {w_support}, AUC {w_auc}.')
        logger.info(f'Average difference between conditional probabilities: {avg_dif}')
        logger.info(f'Average absolute difference between conditional probabilities: {avg_abs_dif}')
    else:
        utils.make_coplot(protected_test_dict, labels_test_dict)

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
    parser.add_argument('--debias', action='store_true',
                        help='Use the adversarial network to mitigate unwanted bias')
    parser.add_argument('--val', action="store_true",
                        help='Use a validation set during training')
    parser.add_argument('--dataset', type=str, default='adult',
                        help='Dataset to be used: adult, crime, images')
    parser.add_argument('--seed', type=int, default=None,
                        help='Train with a fixed seed')
    parser.add_argument('--save_model_to', type=str, default="saved_models/",
                        help='Output path for saved model')


    args = parser.parse_args()

    logger.info('Using configuration {}'.format(vars(args)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Using device {}'.format(device))

    # Set seed if given
    if args.seed:
        torch.manual_seed(args.seed)

    # Load data
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

    train(dataloader_train, dataloader_val, predictor, optimizer_P, criterion, metric, adversary, optimizer_A, scheduler,
          device)
    #test(dataloader_test, predictor, optimizer_P, criterion, metric, adversary, optimizer_A, scheduler, device)
