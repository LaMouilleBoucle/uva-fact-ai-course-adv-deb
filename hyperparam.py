import json
from collections import defaultdict

import argparse
import logging
import os

import coloredlogs
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import math
import numpy as np
from datetime import datetime

import utils
from data.adult_dataset_preprocess import AdultUCI

from sklearn.metrics import accuracy_score

from model import Predictor
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
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_path, 'data/adult.data')
    test_path = os.path.join(base_path, 'data/adult.test')
    data = AdultUCI([train_path, test_path], ['sex'])
    end_of_train = int(0.7*len(data))

    train_dataset = Subset(data, range(0, end_of_train))
    test_dataset = Subset(data, range(end_of_train, len(data)))

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
    optimizer_P = torch.optim.Adam(predictor.parameters(), lr=args.predictor_lr)
    optimizer_A = torch.optim.Adam(adversary.parameters(), lr=args.adversary_lr)

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

    if args.debias:
        # Learning rate decay
        decayer.step_count = 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_P, decayer)

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
            protected_train['true'].extend(true_z_label.cpu().numpy().tolist())
            true_z_label.unsqueeze_(dim=1)

            # forward step predictior
            pred_y_logit, pred_y_label = predictor(x_train)

            # compute loss predictor
            loss_P = criterion(pred_y_label, true_y_label)

            if args.debias:
                # forward step adverserial
                pred_z_logit, pred_z_label = adversary(pred_y_label, true_y_label)

                # compute loss adverserial
                loss_A = criterion(pred_z_label, true_z_label)

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
                alpha = math.sqrt(decayer.step_count)

                # modify and replace the gradient of the predictor
                grad_w_Lp = grad_w_Lp - proj_grad - alpha * grad_w_La
                utils.replace_grad(predictor, grad_w_Lp)


            # update predictor weights
            optimizer_P.step()

            if args.debias:
                # Decay the learning rate
                scheduler.step()

                # Update adversary params
                optimizer_A.step()

                # store train loss and prediction of the adversery
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
        # logger.info('Epoch {}/{}: predictor loss [train] = {:.3f}, '
        #             'predictor accuracy [train] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(train_losses_P), train_accuracy_P))
        train_accuracies_P.append(train_accuracy_P)

        if args.debias:
            av_train_losses_A.append(np.mean(train_losses_A))
            # train_accuracy_A = accuracy_score(train_dataset.z.squeeze(dim=1).numpy(), train_predictions_A)
            train_accuracy_A = accuracy_score(protected_train['true'], protected_train['pred'])
            # logger.info('Epoch {}/{}: adversary loss [train] = {:.3f}, '
            #             'adversary accuracy [train] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(train_losses_A),
            #                                                          train_accuracy_A))
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
            # logger.info('Epoch {}/{}: predictor loss [val] = {:.3f}, '
            #             'predictor accuracy [val] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(val_losses_P), val_accuracy_P))
            val_accuracies_P.append(val_accuracy_P)


            if args.debias:
                av_val_losses_A.append(np.mean(val_losses_A))
                # val_accuracy_A = accuracy_score(val_dataset.z.squeeze(dim=1).numpy(), val_predictions_A)
                val_accuracy_A = accuracy_score(protected_val['true'], protected_val['pred'])
                # logger.info('Epoch {}/{}: adversary loss [val] = {:.3f}, '
                #             'adversary accuracy [val] = {:.3f}'.format(epoch + 1, args.n_epochs, np.mean(val_losses_A),
                                                                       # val_accuracy_A))
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

    # run the model on the test set after training
    with torch.no_grad():

        for i, (x, y, z) in enumerate(dataloader_test):

            x_test = x.to(device)
            true_y_label = y.to(device)
            true_y_label.unsqueeze_(dim=1)
            labels_test['true'].extend(true_y_label.cpu().numpy().tolist())
            true_z_label = z.to(device)
            protected_test['true'].extend(true_z_label.cpu().numpy().tolist())

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
    # logger.info('Generating plots')
    # utils.plot_loss_acc((av_train_losses_P,train_accuracies_P), (av_train_losses_A,train_accuracies_A))

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
    parser.add_argument('--seed', type=int, default=40,
                        help='Seed used to generate other seeds for runs')

    args = parser.parse_args()


    lr_P = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    lr_A = [0.0001, 0.001, 0.01, 0.1, 1]
    batch_size = [32, 64, 128]

    data = defaultdict(lambda: defaultdict(list))


    for p in lr_P:
        for a in lr_A:
            for batch in batch_size:

                key = str((p, a, batch))

                for i in range(10):
                    args.predictor_lr = p
                    args.adversary_lr = a
                    args.batch_size = batch

                    neg_confusion_mat, neg_fpr, neg_fnr, pos_confusion_mat, pos_fpr, pos_fnr, predictor_acc = train(args.seed+i)

                    data[key]["neg_fpr"].append(neg_fpr)
                    data[key]["neg_fnr"].append(neg_fnr)
                    data[key]["pos_fpr"].append(pos_fpr)
                    data[key]["pos_fnr"].append(pos_fnr)
                    data[key]["predictor_acc"].append(predictor_acc)


                    data[key]["neg_confusion_mat"].append(neg_confusion_mat.tolist())
                    data[key]["pos_confusion_mat"].append(pos_confusion_mat.tolist())

    file_name = "data-" + str(datetime.now()).replace(':', '-').replace(' ', '_') + ".json"

    if args.debias:
        file_name = "data_debias-" + str(datetime.now()).replace(':', '-').replace(' ', '_') + ".json"

    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii = False, indent = 4)

    logger.info('Results written to file: {}'.format(file_name))
