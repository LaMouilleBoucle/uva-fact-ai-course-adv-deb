import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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


def project_grad(x, v):
    norm_v = v / (torch.norm(v) + torch.finfo(torch.float32).tiny)
    proj_grad = torch.dot(x, norm_v) * v
    return proj_grad


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
