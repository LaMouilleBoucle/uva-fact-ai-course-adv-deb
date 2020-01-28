import torch

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score



def decayer(lr):
    new_lr = lr / decayer.step_count
    decayer.step_count += 1
    return new_lr


def forward_full(dataloader, predictor, optimizer_P, criterion, adversary, optimizer_A, scheduler, device, dataset, train=False,):
    labels_dict = {'true': [], 'pred': []}
    protected_dict = {'true': [], 'pred': []}
    losses_P, losses_A = [], []

    for i, (x, y, z) in enumerate(dataloader):
        x = x.to(device)
        # print(y.shape)
        true_y_label = y.to(device).unsqueeze_(dim=1)
        true_z_label = z.to(device).unsqueeze_(dim=1)
        # print(y.shape)
        # print(true_y_label.shape)

        # Forward step through predictor

        pred_y_logit, pred_y_prob = predictor(x)

        # Compute loss with respect to predictor
        loss_P = criterion(pred_y_prob, true_y_label)
        losses_P.append(loss_P.item())

        if dataset == 'images':
            labels_dict['true'].extend(torch.max(true_y_label, dim=2)[1].squeeze().cpu().numpy().tolist())
            labels_dict['pred'].extend(torch.max(pred_y_prob, dim=1)[1].cpu().numpy().tolist())
        elif dataset == 'adult':
            labels_dict['true'].extend(y.numpy().tolist())
            pred_y = (pred_y_prob > 0.5).squeeze(dim=1).cpu().numpy().tolist()
            labels_dict['pred'].extend(pred_y)
        else:
            labels_dict['true'].extend(y.numpy().tolist())
            labels_dict['pred'].extend(pred_y_prob.detach().numpy().tolist())
        protected_dict['true'].extend(z.numpy().tolist())

        if adversary is not None:
            # Forward step through adversary
            pred_z_logit, pred_z_prob = adversary(pred_y_logit, true_y_label)

            # Compute loss with respect to adversary
            loss_A = criterion(pred_z_prob, true_z_label)
            losses_A.append(loss_A.item())

            if dataset == 'crime':
                pred_z = pred_z_prob.detach().numpy().tolist()
            else:
                pred_z = (pred_z_prob > 0.5).float().squeeze(dim=1).cpu().numpy().tolist()
            protected_dict['pred'].extend(pred_z)

        if train:

            if adversary is not None:
                # Reset gradients of adversary and predictor
                optimizer_A.zero_grad()
                optimizer_P.zero_grad()
                # Compute gradients adversary loss
                loss_A.backward(retain_graph=True)
                # Concatenate gradients of adversary loss with respect to the predictor
                grad_w_La = concat_grad(predictor)

            # Reset gradients of predictor
            optimizer_P.zero_grad()

            # Compute gradients of predictor loss
            loss_P.backward()

            if adversary is not None:
                # Concatenate gradients of predictor loss with respect to the predictor
                grad_w_Lp = concat_grad(predictor)
                # Project gradients of the predictor
                proj_grad = project_grad(grad_w_Lp, grad_w_La)
                # Set alpha parameter
                alpha = 0.1 # math.sqrt(decayer.step_count)
                # Modify and replace the gradient of the predictor
                grad_w_Lp = grad_w_Lp - proj_grad - alpha * grad_w_La
                replace_grad(predictor, grad_w_Lp)

            # Update predictor weights
            optimizer_P.step()

            if adversary is not None:
                # Decay the learning rate
                # if decayer.step_count % 1000 == 0:
                #     scheduler.step()
                # Update adversary params
                optimizer_A.step()

    return losses_P, losses_A, labels_dict, protected_dict


def concat_grad(model):
    g = None
    for name, param in model.named_parameters():
        grad = param.grad
        if "bias" in name:
            grad = grad.unsqueeze(dim=0)
        if g is None:
            # print(name)
            # print(param.grad.shape)
            g = param.grad.view(1,-1)
            # print(g.shape)
        else:
            if len(grad.shape) < 2:
                grad = grad.unsqueeze(dim=0)
            else:
                grad = grad.view(1, -1)
            g = torch.cat((g, grad), dim=1)
    return g.squeeze(dim=0)


def replace_grad(model, grad):
    start = 0
    for name, param in model.named_parameters():
        numel = param.numel()
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


def calculate_metrics(true_labels, predictions, true_protected, dataset):
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    # print(true_labels.shape)
    # print(predictions.shape)
    # print(true_protected.shape)

    negative_indices = np.where(np.array(true_protected) == 0)[0]
    positive_indices = np.where(np.array(true_protected) == 1)[0]
    neg_confusion_mat = confusion_matrix(true_labels[negative_indices], predictions[negative_indices])
    pos_confusion_mat = confusion_matrix(true_labels[positive_indices], predictions[positive_indices])

    if dataset == 'adult':
        tn, fp, fn, tp = neg_confusion_mat.ravel()
        neg_fpr = calculate_fpr(fp, tn)
        neg_fnr = calculate_fnr(fn, tp)

        tn, fp, fn, tp = pos_confusion_mat.ravel()
        pos_fpr = calculate_fpr(fp, tn)
        pos_fnr = calculate_fnr(fn, tp)

        return neg_confusion_mat, neg_fpr, neg_fnr, pos_confusion_mat, pos_fpr, pos_fnr
    elif dataset == 'images':
        # 0 is male
        neg_conditionals = conditional_matrix(neg_confusion_mat)
        pos_conditionals = conditional_matrix(pos_confusion_mat)
        protected_differences = neg_conditionals - pos_conditionals
        avg_dif = np.average(protected_differences, axis=1)
        avg_abs_dif = np.average(np.absolute(protected_differences), axis=1)
        m_prec, m_recall, m_fscore, m_support = precision_recall_fscore_support(true_labels[negative_indices], predictions[negative_indices])
        w_prec, w_recall, w_fscore, w_support = precision_recall_fscore_support(true_labels[positive_indices], predictions[positive_indices])
        m_acc = neg_confusion_mat.diagonal()/neg_confusion_mat.sum(axis=1)
        w_acc = pos_confusion_mat.diagonal()/pos_confusion_mat.sum(axis=1)

        return m_prec, m_recall, m_fscore, m_support, m_acc, w_prec, w_recall, w_fscore, w_support, w_acc, avg_dif, avg_abs_dif

def conditional_matrix(confusion_matrix):
    # y axis = true label
    # x axis = pred label
    # p(y_hat| y) = p(y_hat, y) / p(y)
    normalization = np.expand_dims(np.sum(confusion_matrix, axis=1), axis=1)
    conditional_matrix = np.divide(confusion_matrix, normalization)
    return conditional_matrix

def plot_loss_acc(P, A=None, dataset='adult'):
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
    if A is None:
        title = f'train_{dataset}_no_debias.png'
    else:
        title = f'train_{dataset}_debias.png'
    plt.savefig(title)
