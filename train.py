import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from collections import defaultdict
import json
import math
import matplotlib.pyplot as plt
import numpy as np

from datasets.toy_dataset import ToyDataset

from sklearn.metrics import accuracy_score

from model import Predictor
from model import Adversary


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # load data
    train_dataset = ToyDataset(1000)
    val_dataset = ToyDataset(1000)
    test_dataset = ToyDataset(1000)

    dataloader_train = DataLoader(train_dataset, args.batch_size)
    dataloader_val = DataLoader(val_dataset, args.batch_size)
    dataloader_test = DataLoader(test_dataset, args.batch_size)

    # get feature dimension of data 
    features_dim = train_dataset.x.shape[1]

    # Initialize models (for toy data the adversary is also logistic regression)
    predictor = Predictor(features_dim).to(device)
    adversary = Predictor(1).to(device)

    # initialize optimizers
    optimizer_P = torch.optim.Adam(predictor.parameters())
    optimizer_A = torch.optim.Adam(adversary.parameters())

    # setup the loss function 
    criterion = nn.BCELoss()


    av_train_losses_P = []
    av_train_losses_A = []
    av_val_losses_P = [] 
    av_val_losses_A = []

    train_accuracies = []
    val_accuracies = []

    # needed for alpha parameter 
    step = 1
    
    for epoch in range(args.n_epochs):


        train_losses_P = [] 
        train_losses_A = [] 
        val_losses_P = [] 
        val_losses_A = []

        train_predictions = []
        val_predictions = []


        for i, (x, y, z) in enumerate(dataloader_train):

        	# maybe something like this is needed to implement the stable learning of predictor
        	# during training on UCI Adult dataset
            # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_P, lambda x: x/step)

            x_train = x.to(device)
            true_y_label = y.to(device)
            true_z_label = z.to(device)

            # forward step predictior
            pred_y_logit, pred_y_label = predictor(x_train)

            # compute loss predictor
            loss_P = criterion(pred_y_label, true_y_label)

           
            if args.debias:
                # forward step adverserial
                pred_z_logit, pred_z_label = adversary(pred_y_label)

                # compute loss adverserial
                loss_A = criterion(pred_z_label, true_z_label)

                # reset gradients adversary
                optimizer_A.zero_grad()
                optimizer_P.zero_grad()

                # compute gradients adversary
                loss_A.backward(retain_graph=True)

                # concatenate gradients of adversary params
                grad_w_La = concat_grad(predictor)


            # reset gradients 
            optimizer_P.zero_grad()

            # compute gradients predictor 
            loss_P.backward()

            if args.debias:

                # concatenate gradients of predictor params
                grad_w_Lp = concat_grad(predictor)

                # project predictor gradients 
                proj_grad = (torch.dot(grad_w_Lp, grad_w_La) / torch.dot(grad_w_La, grad_w_La)) * grad_w_La

                # set alpha parameter 
                alpha = math.sqrt(step)

                # modify and replace the gradient of the predictor 
                grad_w_Lp = grad_w_Lp - proj_grad - alpha * grad_w_La
                replace_grad(predictor, grad_w_Lp)

            # update predictor weights 
            optimizer_P.step()


        	# maybe something like this is needed to implement the stable learning of predictor
        	# during training on UCI Adult dataset
            # scheduler.step()


            if args.debias:
                # update adversary params 
                ####! is done here because IBM implementation does that ... 
                optimizer_A.step()

                # store train loss of adverserial 
                train_losses_A.append(loss_A.item())

            # store train loss and prediction of predictor    
            train_losses_P.append(loss_P.item())
            train_predictions.extend((pred_y_label > 0.5).squeeze(dim=1).cpu().numpy().tolist())

            step += 1


    	# evaluate after every epoch 
        with torch.no_grad():

            for i, (x, y, z) in enumerate(dataloader_val):

                x_val = x.to(device)
                true_y_label = y.to(device)
                true_z_label = z.to(device)

                # forward step predictior
                pred_y_logit, pred_y_label = predictor(x_val)

                # compute loss predictor
                loss_P_val = criterion(pred_y_label, true_y_label)

                if args.debias:
                    # forward step adverserial
                    pred_z_logit, pred_z_label = adversary(pred_y_logit)

                    # compute loss adverserial
                    loss_A_val = criterion(pred_z_label, true_z_label)

                    # store validation loss of adverserial 
                    val_losses_A.append(loss_A_val.item())

                # store validation loss and prediction of predictor 
                val_losses_P.append(loss_P_val.item())
                val_predictions.extend((pred_y_label > 0.5).squeeze(dim=1).cpu().numpy().tolist())
        

        # store average losses of predictor after every epoch    
        av_train_losses_P.append(np.mean(train_losses_P))
        av_val_losses_P.append(np.mean(val_losses_P))

        # store train and validation accuracy of predictor after every epoch 
        train_accuracy = accuracy_score(train_dataset.y.squeeze(dim=1).numpy(), train_predictions)
        val_accuracy = accuracy_score(val_dataset.y.squeeze(dim=1).numpy(), val_predictions)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        

        if args.debias: 
        	# store average losses of predictor after every epoch    
        	av_train_losses_A.append(np.mean(train_losses_A))
        	av_val_losses_A.append(np.mean(val_losses_A))


    # print parameters after training 
    for name, param in predictor.named_parameters():
        print('Name: {}, value: {}'.format(name, param))


    # plot accuracy and loss curves
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(np.arange(1, args.n_epochs +1), av_train_losses_P, label="Train loss predictor", color="#E74C3C")
    axs[0].plot(np.arange(1, args.n_epochs +1), av_val_losses_P, label="Val loss predictor", color="#8E44AD")
    
    axs[2].plot(np.arange(1, args.n_epochs + 1), train_accuracies, label='Train accuracy predictor', color="#229954")
    axs[2].plot(np.arange(1, args.n_epochs + 1), val_accuracies, label='Val accuracy predictor', color="#E67E22")

    if args.debias: 
    	axs[1].plot(np.arange(1, args.n_epochs +1), av_train_losses_A, label="Train loss adversary", color="#3498DB")
    	axs[1].plot(np.arange(1, args.n_epochs +1), av_val_losses_A, label="Val loss adversary", color="#FFC300")


    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend(loc="upper right")

    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc="upper right")

    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Accuracy')
    axs[2].legend(loc="upper right")

    plt.tight_layout()

    plt.show()


def concat_grad(model):

    g = None

    for name, param in model.named_parameters():
        grad = param.grad
        if "bias" in name:
            grad = grad.unsqueeze(dim=1)
        if g is None:
            g = param.grad
        else:
            g = torch.cat((g, grad), dim=1)

    return g.squeeze(dim=0)


def replace_grad(model, grad):

    start = 0

    for name, param in model.named_parameters():
        numel = param.numel()
        param.grad.data = grad[start:start + numel].expand_as(param.grad)
        start += numel

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='number of epochs')

    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')

    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Frequency of evaluation on the validation set')

    parser.add_argument('--print_every', type=int, default=100,
                        help='number of iterations after which the training progress is printed')
    parser.add_argument('--debias',  action="store_true",
                        help='Use the adversial network to mitigate unwanted bias')
    args = parser.parse_args()


    train()
