import torch
import numpy as np
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


def generate_confusion_matrix(true_labels, predictions, true_protected):
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    negative_indices = np.where(np.array(true_protected) == 0)[0]
    neg_confusion_mat = confusion_matrix(true_labels[negative_indices], predictions[negative_indices])
    positive_indices = np.where(np.array(true_protected) == 1)[0]
    pos_confusion_mat = confusion_matrix(true_labels[positive_indices], predictions[positive_indices])
    return neg_confusion_mat, pos_confusion_mat
