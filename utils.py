import torch
from torch.utils.data import DataLoader, Subset, random_split
from data.adult_dataset_preprocess import AdultUCI
from data.image_dataset_preprocess import UTKFace

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MaxAbsScaler

def get_dataloaders(batch_size, logger, images=False):
    base_path = os.path.dirname(os.path.abspath(__file__))

    if images:
        data_path = os.path.join(base_path, 'data/UTKFace')
        data = UTKFace(data_path, protected_vars=['sex'])
        train_data, test_data = torch.utils.data.random_split(data, [math.ceil(len(data)*0.6), len(data) - math.ceil(len(data)*0.6)])
        test_data, val_data = torch.utils.data.random_split(test_data, [math.ceil(len(test_data)*0.5), len(test_data) - math.ceil(len(test_data)*0.5)])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

        logger.info(f'Train set is {len(train_data)} images; val set is {len(val_data)}; test set is {len(test_data)}.')

        return (train_loader, val_loader, test_loader)

    else:
        
        train_path = os.path.join(base_path, 'data/adult.data')
        test_path = os.path.join(base_path, 'data/adult.test')
        data = AdultUCI([train_path, test_path], ['sex'])

        train_dataset = Subset(data, range(0, end_of_train))
        test_dataset = Subset(data, range(end_of_train, len(data)))

        dataloader_train = DataLoader(train_dataset, batch_size, shuffle=True)
        dataloader_test = DataLoader(test_dataset, batch_size, shuffle=True)
        dataloaders = (dataloader_train, dataloader_test)        
        logger.info(f'Train set is {len(train_dataset)} samples; test set is {len(test_dataset)}.')

        return (dataloader_train, dataloader_test)


def decayer(lr):
    new_lr = lr / decayer.step_count
    decayer.step_count += 1
    return new_lr


def forward_full(dataloader, model, logger, train=False):
    predictor, adversary, criterion, optimizer_P, optimizer_A, scheduler, device = model
    labels_dict, protected_dict, losses_lsts, preds_lsts = logger

    for i, batch in enumerate(dataloader):

        labels_dict['true'].extend(batch[1].numpy().tolist())
        protected_dict['true'].extend(batch[2].numpy().tolist())

        losses, predictions = forward_batch(batch, predictor, criterion, adversary, device)
        if train:
            backward_batch(predictor, (optimizer_P, (optimizer_A, scheduler)), losses)

        if len(losses) == 2:
        # store train loss and prediction of the adversery
            losses_lsts[1].append(losses[1].item())
            protecteds = (predictions[1] > 0.5).float().squeeze(dim=1).cpu().numpy().tolist()
            preds_lsts[1].extend(protecteds)
            protected_dict['pred'].extend(protecteds)

        # store train loss and prediction of predictor
        losses_lsts[0].append(losses[0].item())
        preds = (predictions[0] > 0.5).squeeze(dim=1).cpu().numpy().tolist()
        preds_lsts[0].extend(preds)
        labels_dict['pred'].extend(preds)

def forward_batch(batch, predictor, criterion, adversary, device):
    x, y, z = batch
    
    x_train = x.to(device)
    true_y_label = y.to(device).unsqueeze_(dim=1)
    true_z_label = z.to(device).unsqueeze_(dim=1)

    # forward step predictior
    # pred_y_logit, pred_y_label = predictor(x_train)
    pred_y_label = predictor(x_train)

    # compute loss predictor
    loss_P = criterion(pred_y_label, true_y_label)

    if adversary is not None:
        # forward step adverserial
        pred_z_logit, pred_z_label = adversary(pred_y_label, true_y_label)

        # compute loss adverserial
        loss_A = criterion(pred_z_label, true_z_label)

        return [loss_P, loss_A], [pred_y_label, pred_z_label]

    return [loss_P], [pred_y_label]


def backward_batch(predictor, optimizers, losses):
    debias = len(losses) == 2

    if debias:
        # reset gradients adversary
        optimizers[1][0].zero_grad()
        optimizers[0].zero_grad()
        # compute gradients adversary
        losses[1].backward(retain_graph=True)
        # concatenate gradients of adversary params
        grad_w_La = concat_grad(predictor)

    # reset gradients
    optimizers[0].zero_grad()
    # compute gradients predictor
    losses[0].backward()

    if debias:
        # concatenate gradients of predictor params
        grad_w_Lp = concat_grad(predictor)
        # project predictor gradients
        proj_grad = project_grad(grad_w_Lp, grad_w_La)
        # set alpha parameter
        alpha = math.sqrt(decayer.step_count)
        # modify and replace the gradient of the predictor
        grad_w_Lp = grad_w_Lp - proj_grad - alpha * grad_w_La
        replace_grad(predictor, grad_w_Lp)

    # update predictor weights
    optimizers[0].step()

    if debias:
        # Decay the learning rate
        optimizers[1][1].step()
        # Update adversary params
        optimizers[1][0].step()


def concat_grad(model):
    g = None
    for name, param in model.named_parameters():
        grad = param.grad
        if "bias" in name:
            grad = grad.unsqueeze(dim=0)
        if g is None:
            g = param.grad.view(1,-1)
        else:
            g = torch.cat((g, grad), dim=1)
    return g.squeeze(dim=0)


def replace_grad(model, grad):
    start = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        # print(name)
        # print(param.grad.shape)
        # print(grad[start:start + numel].shape)
        # print('')
        param.grad.data = grad[start:start + numel].view(param.grad.shape)
        start += numel


def project_grad(x, v):
    norm_v = v / (torch.norm(v) + torch.finfo(torch.float32).tiny)
    proj_grad = torch.dot(x, norm_v) * v
    return proj_grad


def calculate_fpr(fp, tn):
    return fp / (fp + tn)


def calculate_fnr(fn, tp):
    return fn / (fn + tp)


def calculate_metrics(true_labels, predictions, true_protected):
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    negative_indices = np.where(np.array(true_protected) == 0)[0]
    neg_confusion_mat = confusion_matrix(true_labels[negative_indices], predictions[negative_indices])
    tn, fp, fn, tp = neg_confusion_mat.ravel()
    neg_fpr = calculate_fpr(fp, tn)
    neg_fnr = calculate_fnr(fn, tp)

    positive_indices = np.where(np.array(true_protected) == 1)[0]
    pos_confusion_mat = confusion_matrix(true_labels[positive_indices], predictions[positive_indices])
    tn, fp, fn, tp = pos_confusion_mat.ravel()
    pos_fpr = calculate_fpr(fp, tn)
    pos_fnr = calculate_fnr(fn, tp)

    return neg_confusion_mat, neg_fpr, neg_fnr, pos_confusion_mat, pos_fpr, pos_fnr


def plot_loss_acc(P, A=None):
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(np.arange(1, len(P[0])+1), P[0], label="Train loss predictor", color="#E74C3C")
    #axs[0].plot(np.arange(1, args.n_epochs +1), av_val_losses_P, label="Val loss predictor", color="#8E44AD")

    axs[2].plot(np.arange(1, len(P[1])+1), P[1], label='Train accuracy predictor', color="#229954")
    #axs[2].plot(np.arange(1, args.n_epochs + 1), val_accuracies_P, label='Val accuracy predictor', color="#E67E22")

    if A is not None:
        axs[1].plot(np.arange(1, len(A[0])+1), A[0], label="Train loss adversary", color="#3498DB")
        #axs[1].plot(np.arange(1, args.n_epochs +1), av_val_losses_A, label="Val loss adversary", color="#FFC300")

        axs[3].plot(np.arange(1, len(A[1])+1), A[1], label='Train accuracy adversary', color="#229954")
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
