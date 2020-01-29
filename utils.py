import torch

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_auc_score
from collections import Counter
from collections import defaultdict



def decayer(lr):
    new_lr = lr / decayer.step_count
    decayer.step_count += 1
    return new_lr


def forward_full(dataloader, predictor, optimizer_P, criterion, adversary, optimizer_A, scheduler, device, dataset,
                 train=False):
    labels_dict = {'true': [], 'pred': []}
    protected_dict = {'true': [], 'pred': []}
    losses_P, losses_A = [], []
    # prediction_probs = []

    for i, (x, y, z) in enumerate(dataloader):

        x = x.to(device)
        true_y_label = y.to(device)
        true_z_label = z.to(device)

        # Forward step through predictor
        pred_y_logit, pred_y_prob = predictor(x)

        if train is False:
            if i == 0:
                prediction_probs = pred_y_prob.cpu().detach().numpy()
            else:
                prediction_probs = np.concatenate((prediction_probs, pred_y_prob.cpu().detach().numpy()), axis=0)

        # Compute loss with respect to predictor
        loss_P = criterion(pred_y_prob, true_y_label)
        losses_P.append(loss_P.item())

        if dataset == 'images':
            labels_dict['true'].extend(torch.max(true_y_label, dim=1)[1].cpu().numpy().tolist())
            labels_dict['pred'].extend(torch.max(pred_y_prob, dim=1)[1].cpu().numpy().tolist())
        elif dataset == 'adult':
            labels_dict['true'].extend(y.squeeze().numpy().tolist())
            pred_y = (pred_y_prob > 0.5).int().squeeze(dim=1).cpu().numpy().tolist()
            labels_dict['pred'].extend(pred_y)
        else:
            labels_dict['true'].extend(y.numpy().tolist())
            labels_dict['pred'].extend(pred_y_prob.detach().numpy().tolist())
        protected_dict['true'].extend(z.squeeze().numpy().tolist())

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
                alpha = 0.3 # math.sqrt(decayer.step_count)
                # Modify and replace the gradient of the predictor
                grad_w_Lp = grad_w_Lp - proj_grad - alpha * grad_w_La
                replace_grad(predictor, grad_w_Lp)

            # Update predictor weights
            optimizer_P.step()
            decayer.step_count += 1

            if adversary is not None:
                # Decay the learning rate
                if decayer.step_count % 1000 == 0:
                    scheduler.step()
                # Update adversary params
                optimizer_A.step()

    if train:
        return losses_P, losses_A, labels_dict, protected_dict
    else:
        mutual_info = mutual_information(protected_dict["true"], labels_dict['true'], labels_dict['pred'])
        return losses_P, losses_A, labels_dict, protected_dict, prediction_probs, mutual_info


def concat_grad(model):
    """
    Concatenates the gradients of a model to form a single parameter vector tensor.

    Args:
        model (nn.Module): PyTorch model object

    Returns: A single vector tensor with the model gradients concatenated
    """
    g = None
    for name, param in model.named_parameters():
        grad = param.grad
        if "bias" in name:
            grad = grad.unsqueeze(dim=0)
        if g is None:
            g = param.grad.view(1, -1)
        else:
            if len(grad.shape) < 2:
                grad = grad.unsqueeze(dim=0)
            else:
                grad = grad.view(1, -1)
            g = torch.cat((g, grad), dim=1)
    return g.squeeze(dim=0)


def replace_grad(model, grad):
    """
    Replace the gradients of the model with the specified gradient vector tensor

    Args:
        model (nn.Module): PyTorch model object
        grad (Tensor): Vector of concatenated gradients

    Returns: None
    """
    start = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        param.grad.data = grad[start:start + numel].view(param.grad.shape)
        start += numel


def project_grad(x, v):
    """
    Performs a projection of one vector on another

    Args:
        x (Tensor): Vector to project
        v (Tensor): Vector on which projection is required

    Returns: Tensor containing the projected vector
    """
    norm_v = v / (torch.norm(v) + torch.finfo(torch.float32).tiny)
    proj_grad = torch.dot(x, norm_v) * v
    return proj_grad


def calculate_fpr(fp, tn):
    """
    Calculates false positive rate (FPR)

    Args:
        fp (int): Number of false positives
        tn (int): Number of true negatives

    Returns: False positive rate
    """
    return fp / (fp + tn)


def calculate_fnr(fn, tp):
    """
    Calculates false negative rate (FNR)

    Args:
        fn (int): Number of false negatives
        tp (int): Number of true positives

    Returns: False negative rate
    """
    return fn / (fn + tp)


def calculate_metrics(true_labels, predictions, true_protected, dataset, pred_probs=None):
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

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
        # 0 is male, so negative = male; positive = female
        neg_conditionals = conditional_matrix(neg_confusion_mat)
        pos_conditionals = conditional_matrix(pos_confusion_mat)
        protected_differences = neg_conditionals - pos_conditionals
        avg_dif = np.average(protected_differences, axis=1)
        avg_abs_dif = np.average(np.absolute(protected_differences), axis=1)
        neg_prec, neg_recall, neg_fscore, neg_support = precision_recall_fscore_support(true_labels[negative_indices], 
                                                                                        predictions[negative_indices])
        pos_prec, pos_recall, pos_fscore, pos_support = precision_recall_fscore_support(true_labels[positive_indices], 
                                                                                        predictions[positive_indices])
        if pred_probs is not None:
            one_hot_labels = np.zeros((true_labels.size, true_labels.max()+1))
            one_hot_labels[np.arange(true_labels.size),true_labels] = 1
            neg_auc = roc_auc_score(one_hot_labels[negative_indices], pred_probs[negative_indices])
            pos_auc = roc_auc_score(one_hot_labels[positive_indices], pred_probs[positive_indices])

        return neg_prec, neg_recall, neg_fscore, neg_support, neg_auc, pos_prec, pos_recall, pos_fscore, pos_support, pos_auc, avg_dif, avg_abs_dif


def conditional_matrix(confusion_matrix):
    # y axis = true label
    # x axis = pred label
    # p(y_hat| y) = p(y_hat, y) / p(y)
    normalization = np.expand_dims(np.sum(confusion_matrix, axis=1), axis=1)
    conditional_matrix = np.divide(confusion_matrix, normalization)
    return conditional_matrix


def plot_loss_acc(P, A=None, dataset='adult'):
    fig, axs = plt.subplots(4, 1)
    axs[0].plot(np.arange(1, len(P[0]) + 1), P[0], label="Train loss predictor", color="#E74C3C")
    # axs[0].plot(np.arange(1, args.n_epochs +1), av_val_losses_P, label="Val loss predictor", color="#8E44AD")

    axs[2].plot(np.arange(1, len(P[1]) + 1), P[1], label='Train accuracy predictor', color="#229954")
    # axs[2].plot(np.arange(1, args.n_epochs + 1), val_accuracies_P, label='Val accuracy predictor', color="#E67E22")

    if A is not None:
        axs[1].plot(np.arange(1, len(A[0]) + 1), A[0], label="Train loss adversary", color="#3498DB")
        # axs[1].plot(np.arange(1, args.n_epochs +1), av_val_losses_A, label="Val loss adversary", color="#FFC300")

        axs[3].plot(np.arange(1, len(A[1]) + 1), A[1], label='Train accuracy adversary', color="#229954")
        # axs[3].plot(np.arange(1, args.n_epochs + 1), val_accuracies_A, label='Val accuracy adversary', color="#E67E22")

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


def entropy(rv1, cond_rv):
    entropy = 0

    if cond_rv is None:
        # Calculate entropy H(rv1)
        distr_rv1 = get_distr(rv1)
        for prob in distr_rv1.values():
            entropy += prob * math.log(1 / prob, 2)
    else: 
        # Calculate entropy H(rv1 | cond_rv)
        distr_rv1_conditioned = get_conditional_distr(rv1, cond_rv)
        distr_cond_rv = get_distr(cond_rv)

        for event, prob in distr_cond_rv.items(): 
            entropy_part = 0
            for cond_prob in distr_rv1_conditioned[event].values(): 
                entropy_part += cond_prob * math.log(1 / cond_prob, 2)
            entropy += prob * entropy_part
    return entropy


def get_joint(rv1, rv2): 
    return [i for i in zip(rv1, rv2)]


def get_distr(rv): 

    distr = {}
    for event, frequency in Counter(rv).items():
        distr[event] = frequency / len(rv)

    return distr

def get_conditional_distr(rv1, rv2): 

    # Get joint distribution
    distr_joint = get_distr(get_joint(rv1, rv2))

    # Get distribution of random variable we want to condition on 
    distr_rv2 = get_distr(rv2)

    # Get distribution of random variable 1 conditioned of random variable 2
    distr_cond = defaultdict(lambda: {})
    for event, prob in distr_joint.items():

        distr_cond[event[1]][event[0]] = prob / distr_rv2[event[1]]

    return distr_cond



def mutual_information(rv1, rv2, cond_rv = None):

    mutual_information = None

    if cond_rv is None: 

        # Compute entropy H(rv1)
        entropy1 = entropy(rv1)

        # Compute entropy H(rv1 | rv2)
        entropy2 = entropy(rv1, cond_rv = rv2)

        # Compute mutual information I(rv1; rv2)
        mutual_information = entropy1 - entropy2

    else: 
        # Compute entropy H(rv1 | cond_rv)
        entropy1 = entropy(rv1, cond_rv = cond_rv)

        # Compute entropy H(rv1 | cond_rv, rv2)
        entropy2 = entropy(rv1, cond_rv = get_joint(cond_rv, rv2))

        # # Compute mutual information I(rv1; rv2 | cond_rv)
        mutual_information = entropy1 - entropy2

    
    print("MUTUAL INFORMATION", mutual_information)
    return mutual_information


    

