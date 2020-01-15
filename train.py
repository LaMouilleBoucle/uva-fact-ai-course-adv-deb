import argparse
import logging

import coloredlogs
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import math
import matplotlib.pyplot as plt
import numpy as np

import utils
from data.adult_dataset_preprocess import AdultUCI

from datasets.toy_dataset import ToyDataset

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler

from model import Predictor
from model import Adversary

logger = logging.getLogger('Training log')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def stepper(x):
    stepper.step_count += 1
    print('step', stepper.step_count)
    return x * stepper.step_count


def train():
    logger.info('Using configuration {}'.format(vars(args)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Using device {}'.format(device))

    # load data
    logger.info('Loading the dataset')
    data = AdultUCI(['./data/adult.data', './data/adult.test'], ['sex'])
    train_dataset = Subset(data, range(0, data.lengths[0]))
    test_dataset = Subset(data, range(data.lengths[0], data.lengths[0] + data.lengths[1]))

    # Scale each feature by its maximum absolute value
    min_max_scaler = MaxAbsScaler()
    train_dataset.dataset.data = torch.tensor(min_max_scaler.fit_transform(train_dataset.dataset.data.numpy()))
    test_dataset.dataset.data = torch.tensor(min_max_scaler.transform(test_dataset.dataset.data.numpy()))

    dataloader_train = DataLoader(train_dataset, args.batch_size, shuffle=True)
    dataloader_test = DataLoader(test_dataset, args.batch_size, shuffle=True)
    logger.info('Finished loading the dataset')

    # get feature dimension of data
    features_dim = train_dataset.dataset.data.shape[1]

    # Initialize models (for toy data the adversary is also logistic regression)
    predictor = Predictor(features_dim).to(device)
    adversary = Adversary().to(device)
    logger.info('Initialized the predictor and the adversary')

    # initialize optimizers
    optimizer_P = torch.optim.Adam(predictor.parameters(), lr=args.lr)
    optimizer_A = torch.optim.Adam(adversary.parameters(), lr=args.lr)

    # setup the loss function
    criterion = nn.BCELoss()

    av_train_losses_P = []
    av_train_losses_A = []
    av_val_losses_P = []
    av_val_losses_A = []


    train_accuracies_P = []
    train_accuracies_A = []
    val_accuracies_P = []
    val_accuracies_A = []

    # needed for alpha parameter
    stepper.step_count = 1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_P, stepper)

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

        # Reinitializing optimizer to update the learning rate
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_P, lambda x: x/step)

        for i, (x, y, z) in enumerate(dataloader_train):

            # maybe something like this is needed to implement the stable learning of predictor
            # during training on UCI Adult dataset
            # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_P, lambda x: x/step)

            x_train = x.to(device)
            true_y_label = y.to(device)
            labels_train['true'].extend(true_y_label.cpu().numpy().tolist())
            true_y_label.unsqueeze_(dim=1)

            true_z_label = z.to(device)
            protected_train['true'].extend(true_z_label.argmax(dim=1).cpu().numpy().tolist())
            true_z_label.unsqueeze_(dim=2)

            # forward step predictior
            pred_y_logit, pred_y_label = predictor(x_train)

            # compute loss predictor
            loss_P = criterion(pred_y_label, true_y_label)

            if args.debias:
                # forward step adverserial
                pred_z_logit, pred_z_label = adversary(pred_y_label, true_y_label)

                # compute loss adverserial
                loss_A = criterion(pred_z_label, true_z_label.argmax(dim=1).float())

                # reset gradients adversary
                optimizer_A.zero_grad()
                optimizer_P.zero_grad()

                # compute gradients adversary
                loss_A.backward(retain_graph=True)

                # concatenate gradients of adversary params
                grad_w_La = utils.concat_grad(predictor)

            # reset gradients
            optimizer_P.zero_grad()

            # compute gradients predictor
            loss_P.backward()

            if args.debias:

                # concatenate gradients of predictor params
                grad_w_Lp = utils.concat_grad(predictor)

                # project predictor gradients
                proj_grad = utils.project_grad(grad_w_Lp, grad_w_La)

                # set alpha parameter
                alpha = math.sqrt(stepper.step_count)

                # print(grad_w_Lp.norm())
                # print(proj_grad.norm())
                # print(grad_w_La.norm())
                # modify and replace the gradient of the predictor
                grad_w_Lp = grad_w_Lp - proj_grad - alpha * grad_w_La
                utils.replace_grad(predictor, grad_w_Lp)
                # print(grad_w_Lp.norm())
                # print('')


            # update predictor weights
            optimizer_P.step()
            scheduler.step()

            if args.debias:
                # update adversary params
                ####! is done here because IBM implementation does that ...
                optimizer_A.step()

                # store train loss and prediction of adverserial
                train_losses_A.append(loss_A.item())
                protecteds = (pred_z_label > 0.5).float().squeeze(dim=1).cpu().numpy().tolist()
                train_predictions_A.extend(protecteds)
                protected_train['pred'].extend(protecteds)

            # store train loss and prediction of predictor
            train_losses_P.append(loss_P.item())
            preds = (pred_y_label > 0.5).squeeze(dim=1).cpu().numpy().tolist()
            train_predictions_P.extend(preds)
            labels_train['pred'].extend(preds)

        # store average training losses of predictor after every epoch
        av_train_losses_P.append(np.mean(train_losses_P))

        # store train accuracy of predictor after every epoch
        train_accuracy_P = accuracy_score(labels_train['true'], labels_train['pred'])
        logger.info('Epoch {}/{}: predictor loss [train] = {:.3f}, '
                    'predictor accuracy [train] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(train_losses_P), train_accuracy_P))
        train_accuracies_P.append(train_accuracy_P)

        if args.debias:
            av_train_losses_A.append(np.mean(train_losses_A))
            # train_accuracy_A = accuracy_score(train_dataset.z.squeeze(dim=1).numpy(), train_predictions_A)
            train_accuracy_A = accuracy_score(protected_train['true'], protected_train['pred'])
            logger.info('Epoch {}/{}: adversary loss [train] = {:.3f}, '
                        'adversary accuracy [train] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(train_losses_A),
                                                                     train_accuracy_A))
            train_accuracies_A.append(train_accuracy_A)

        if args.val:

            # evaluate after every epoch
            with torch.no_grad():

                for i, (x, y, z) in enumerate(dataloader_val):

                    x_val = x.to(device)
                    true_y_label = y.to(device)
                    labels_val['true'].extend(true_y_label.cpu().numpy().tolist())
                    true_z_label = z.to(device)
                    protected_val['true'].extend(true_z_label.cpu().numpy().tolist())

                    # forward step predictior
                    pred_y_logit, pred_y_label = predictor(x_val)

                    # compute loss predictor
                    loss_P_val = criterion(pred_y_label, true_y_label)

                    if args.debias:
                        # forward step adverserial
                        pred_z_logit, pred_z_label = adversary(pred_y_logit, true_y_label)

                        # compute loss adverserial
                        loss_A_val = criterion(pred_z_label, true_z_label)

                        # store validation loss of adverserial
                        val_losses_A.append(loss_A_val.item())
                        protecteds = (pred_z_label > 0.5).squeeze(dim=1).cpu().numpy().tolist()
                        val_predictions_A.extend(protecteds)
                        protected_val['pred'].extend(protecteds)

                    # store validation loss and prediction of predictor
                    val_losses_P.append(loss_P_val.item())
                    preds = (pred_y_label > 0.5).squeeze(dim=1).cpu().numpy().tolist()
                    val_predictions_P.extend(preds)
                    labels_val['pred'].extend(preds)


            # store average validation losses of predictor after every epoch
            av_val_losses_P.append(np.mean(val_losses_P))


            # store train accuracy of predictor after every epoch
            # val_accuracy = accuracy_score(val_dataset.labels.squeeze(dim=1).numpy(), val_predictions)
            # val_accuracy_P = accuracy_score(val_dataset.y.squeeze(dim=1).numpy(), val_predictions_P)
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

        # maybe something like this is needed to implement the stable learning of predictor
        # during training on UCI Adult dataset
        # scheduler.step()

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

    # run the model on the test set after training
    with torch.no_grad():

        for i, (x, y, z) in enumerate(dataloader_test):

            x_test = x.to(device)
            true_y_label = y.to(device)
            true_y_label.unsqueeze_(dim=1)
            labels_test['true'].extend(true_y_label.cpu().numpy().tolist())
            true_z_label = z.to(device)
            protected_test['true'].extend(true_z_label.argmax(dim=1).cpu().numpy().tolist())

            # forward step predictior
            pred_y_logit, pred_y_label = predictor(x_test)

            if args.debias:
                # forward step adverserial
                pred_z_logit, pred_z_label = adversary(pred_y_logit, true_y_label)

                # store predictions of predictor and adversary
                protecteds = (pred_z_label > 0.5).float().squeeze(dim=1).cpu().numpy().tolist()
                test_predictions_A.extend(protecteds)
                protected_test['pred'].extend(protecteds)

            #store predictions of predictor and adversary
            preds = (pred_y_label > 0.5).squeeze(dim=1).cpu().numpy().tolist()
            test_predictions_P.extend(preds)
            labels_test['pred'].extend(preds)

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
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(np.arange(1, args.n_epochs +1), av_train_losses_P, label="Train loss predictor", color="#E74C3C")
    #axs[0].plot(np.arange(1, args.n_epochs +1), av_val_losses_P, label="Val loss predictor", color="#8E44AD")

    axs[2].plot(np.arange(1, args.n_epochs + 1), train_accuracies_P, label='Train accuracy predictor', color="#229954")
    #axs[2].plot(np.arange(1, args.n_epochs + 1), val_accuracies_P, label='Val accuracy predictor', color="#E67E22")

    if args.debias:
        axs[1].plot(np.arange(1, args.n_epochs +1), av_train_losses_A, label="Train loss adversary", color="#3498DB")
        #axs[1].plot(np.arange(1, args.n_epochs +1), av_val_losses_A, label="Val loss adversary", color="#FFC300")

        axs[3].plot(np.arange(1, args.n_epochs + 1), train_accuracies_A, label='Train accuracy adversary', color="#229954")
        #axs[3].plot(np.arange(1, args.n_epochs + 1), val_accuracies_A, label='Val accuracy adversary', color="#E67E22")

    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend(loc="upper right")

    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc="upper right")

    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Accuracy')
    axs[2].legend(loc="upper right")

    axs[3].set_xlabel('Epochs')
    axs[3].set_ylabel('Accuracy')
    axs[3].legend(loc="upper right")

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Frequency of evaluation on the validation set')
    parser.add_argument('--print_every', type=int, default=100,
                        help='number of iterations after which the training progress is printed')
    parser.add_argument('--debias',  action='store_true',
                        help='Use the adversial network to mitigate unwanted bias')
    parser.add_argument('--val',  action="store_true",
                        help='Use a validation set during training')

    args = parser.parse_args()

    train()
