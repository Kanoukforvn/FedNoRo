import copy
import random
import os
import sys
import shutil
import numpy as np
import pandas as pd
import logging
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score


import numpy as np
import pandas as pd
import copy
import logging
from typing import List, Dict

def subclasses_transmat(subclass_size: int, first_class: int) -> Dict[int, int]:
    """Make sub-transition matrix of current subclass group for CIFAR-10 asymmetric label noise.

    Args:
        subclass_size (int): Number of subclasses
        first_class (int): Index for the first class for the current subclass group.

    Returns:
        Dict[int, int]: Sub-transition matrix for the current subclass group.
    """
    sub_trans_mat = {}
    for i in range(subclass_size - 1):
        sub_trans_mat[first_class + i] = first_class + i + 1
    sub_trans_mat[first_class + subclass_size - 1] = first_class
    return sub_trans_mat

def build_cifar10_transmat() -> Dict[int, int]:
    nb_classes = 10
    nb_superclasses = 2  # CIFAR-10 has 2 superclasses
    nb_subclasses = 5   # Assuming each superclass has 5 subclasses
    trans_mat = dict()
    for i in range(nb_superclasses):
        init, end = i * nb_subclasses, (i + 1) * nb_subclasses
        sub_trans_mat = subclasses_transmat(nb_subclasses, init)
        trans_mat.update(sub_trans_mat)
    return trans_mat

def asymmetric_label_flipping(
    labels: List[int],
    noise_ratio: float = 0.1,
    transition_matrix: Dict[int, int] = None,
) -> List[int]:
    """Asymmetric label flipping using a transition matrix."""
    sample_num = len(labels)
    idxs = list(range(sample_num))
    random.shuffle(idxs)
    num_noise = int(sample_num * noise_ratio)
    noisy_idxs = idxs[:num_noise]  # indices of candidate noisy sample
    noisy_labels = []

    for i in range(sample_num):
        if i in noisy_idxs:
            noisy_labels.append(transition_matrix[labels[i]])
        else:
            noisy_labels.append(labels[i])

    return noisy_labels

def add_noise(args, y_train, dict_users):
    np.random.seed(args.seed)
    gamma_s = np.array([0.] * args.num_users)
    gamma_s[:int(args.level_n_system * args.num_users)] = 1.
    np.random.shuffle(gamma_s)
    gamma_c_initial = np.random.rand(args.num_users)
    gamma_c_initial = (args.level_n_upperb - args.level_n_lowerb) * \
                      gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial
    y_train_noisy = copy.deepcopy(y_train)

    if args.n_type == "instance":
        if args.dataset == "isic2019":
            df = pd.read_csv("your csv")
        elif args.dataset == "ICH":
            df = pd.read_csv("your csv")
        else:
            raise

        soft_label = df.iloc[:, 1:args.n_classes + 1].values.astype("float")
        real_noise_level = np.zeros(args.num_users)
        for i in np.where(gamma_c > 0)[0]:
            sample_idx = np.array(list(dict_users[i]))
            soft_label_this_client = soft_label[sample_idx]
            hard_label_this_client = y_train[sample_idx]

            p_t = copy.deepcopy(soft_label_this_client[np.arange(
                soft_label_this_client.shape[0]), hard_label_this_client])
            p_f = 1 - p_t
            p_f = p_f / p_f.sum()
            # Choose noisy samples based on the misclassification probability.
            noisy_idx = np.random.choice(np.arange(len(sample_idx)), size=int(
                gamma_c[i] * len(sample_idx)), replace=True, p=p_f)

            for j in noisy_idx:
                soft_label_this_client[j][hard_label_this_client[j]] = 0.
                soft_label_this_client[j] = soft_label_this_client[j] / \
                                            soft_label_this_client[j].sum()
                # Choose a noisy label based on the classification probability.
                # The noisy label is different from the initial label.
                y_train_noisy[sample_idx[j]] = np.random.choice(
                    np.arange(args.n_classes), p=soft_label_this_client[j])

            noise_ratio = np.mean(
                y_train[sample_idx] != y_train_noisy[sample_idx])
            logging.info("Client %d, noise level: %.4f, real noise ratio: %.4f" % (
                i, gamma_c[i], noise_ratio))
            real_noise_level[i] = noise_ratio

    elif args.n_type == "random":
        real_noise_level = np.zeros(args.num_users)
        for i in np.where(gamma_c > 0)[0]:
            sample_idx = np.array(list(dict_users[i]))
            prob = np.random.rand(len(sample_idx))
            noisy_idx = np.where(prob <= gamma_c[i])[0]
            y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(
                0, args.n_classes, len(noisy_idx))
            noise_ratio = np.mean(
                y_train[sample_idx] != y_train_noisy[sample_idx])
            logging.info("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (
                i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio))
            real_noise_level[i] = noise_ratio

    elif args.n_type == "asymmetric_v2":
        real_noise_level = np.zeros(args.num_users)
        for i in np.where(gamma_c > 0)[0]:
            sample_idx = np.array(list(dict_users[i]))

            # Ensure sample_idx does not exceed the size of y_train
            if len(sample_idx) > len(y_train):
                sample_idx = np.random.choice(y_train, size=len(y_train), replace=False)

            num_noise = min(int(len(sample_idx) * gamma_c[i]), len(sample_idx))  # Ensure num_noise is within bounds

            # Randomly select indices for noisy samples
            noisy_idx = np.random.choice(len(sample_idx), size=num_noise, replace=False)

            # Generate noisy labels only for noisy samples
            noisy_labels = asymmetric_label_flipping(
                y_train[sample_idx[noisy_idx]], 
                noise_ratio=gamma_c[i],
                transition_matrix=build_cifar10_transmat()
            )

            # Update y_train_noisy with the noisy labels
            for idx, label in zip(noisy_idx, noisy_labels):
                y_train_noisy[sample_idx[idx]] = label

            noise_ratio = np.mean(
                y_train[sample_idx] != y_train_noisy[sample_idx])
            logging.info("Client %d, noise level: %.4f, real noise ratio: %.4f" % (
                i, gamma_c[i], noise_ratio))
            real_noise_level[i] = noise_ratio


    elif args.n_type == "symmetric":  # Add symmetric noise here
        real_noise_level = np.zeros(args.num_users)
        for i in np.where(gamma_c > 0)[0]:
            sample_idx = np.array(list(dict_users[i]))
            prob = np.random.rand(len(sample_idx))
            noisy_idx = np.where(prob <= gamma_c[i])[0]
            # Introduce symmetric noise by flipping labels randomly
            for idx in noisy_idx:
                correct_label = y_train[sample_idx[idx]]
                other_labels = np.delete(np.arange(args.n_classes), correct_label)
                noisy_label = np.random.choice(other_labels)
                y_train_noisy[sample_idx[idx]] = noisy_label
            noise_ratio = np.mean(
                y_train[sample_idx] != y_train_noisy[sample_idx])
            logging.info("Client %d, noise level: %.4f, real noise ratio: %.4f" % (
                i, gamma_c[i], noise_ratio))
            real_noise_level[i] = noise_ratio

    else:
        raise NotImplementedError

    return y_train_noisy, gamma_s, real_noise_level


def sigmoid_rampup(current, begin, end):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    current = np.clip(current, begin, end)
    phase = 1.0 - (current-begin) / (end-begin)
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(rnd, begin, end):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sigmoid_rampup(rnd, begin, end)


def get_output(loader, net, args, softmax=False, criterion=None):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()
            if softmax == True:
                outputs = net(images)
                outputs = F.softmax(outputs, dim=1)
            else:
                outputs = net(images)
            if criterion is not None:
                loss = criterion(outputs, labels)
            if i == 0:
                output_whole = np.array(outputs.cpu())
                if criterion is not None:
                    loss_whole = np.array(loss.cpu())
            else:
                output_whole = np.concatenate(
                    (output_whole, outputs.cpu()), axis=0)
                if criterion is not None:
                    loss_whole = np.concatenate(
                        (loss_whole, loss.cpu()), axis=0)
    if criterion is not None:
        return output_whole, loss_whole
    else:
        return output_whole


def get_output_and_label(loader, net, args):
    net.eval()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(args.device)
            labels = labels.to(args.device)
            labels = labels.long()

            outputs = net(images)
            outputs = F.softmax(outputs, dim=1)

            if i == 0:
                output_whole = np.array(outputs.cpu())
                label_whole = np.array(labels.cpu())
            else:
                output_whole = np.concatenate(
                    (output_whole, outputs.cpu()), axis=0)
                label_whole = np.concatenate(
                    (label_whole, labels.cpu()), axis=0)

    return output_whole, label_whole


def cal_training_acc(prediction, noisy_labels, true_labels):
    prediction = np.array(prediction)
    noisy_labels = np.array(noisy_labels)
    true_labels = np.array(true_labels)

    acc_noisy = balanced_accuracy_score(noisy_labels, prediction)
    acc_true = balanced_accuracy_score(true_labels, prediction)

    return acc_noisy, acc_true


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_output_files(args):
    outputs_dir = 'outputs_' + str(args.dataset) + '_' + str(
        args.level_n_system) + '_' + str(args.level_n_lowerb) + '_' + str(args.level_n_upperb)
    if not os.path.exists(outputs_dir):
        os.mkdir(outputs_dir)
    exp_dir = os.path.join(outputs_dir, args.exp + '_' +
                           str(args.level_n_system) + '_' + str(args.level_n_lowerb) + '_' +
                           str(args.level_n_upperb) + '_' + str(args.local_ep))
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    models_dir = os.path.join(exp_dir, 'models')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    logs_dir = os.path.join(exp_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    tensorboard_dir = os.path.join(exp_dir, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    code_dir = os.path.join(exp_dir, 'code')
    if os.path.exists(code_dir):
        shutil.rmtree(code_dir)
    # shutil.copytree('.', code_dir, ignore=shutil.ignore_patterns('.git'))

    logging.basicConfig(filename=logs_dir+'/logs.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    writer = SummaryWriter(tensorboard_dir)
    return writer, models_dir

def check_noise_type(labels, noisy_labels):
    """
    Check if the correct type of noise has been generated.
    
    Args:
    - labels (numpy array): Array of original labels.
    - noisy_labels (numpy array): Array of labels after adding noise.
    
    Returns:
    - str: Type of noise detected ("symmetric", "asymmetric", or "unknown").
    """
    # Count occurrences of each label in original and noisy labels
    label_counts = np.bincount(labels)
    noisy_label_counts = np.bincount(noisy_labels)
    
    # Calculate the absolute difference between label counts
    diff = np.abs(label_counts - noisy_label_counts)
    
    # If the sum of differences is zero, it indicates symmetric noise
    if np.sum(diff) == 0:
        return "symmetric"
    
    # If the sum of differences is equal to the maximum difference, it indicates asymmetric noise
    elif np.sum(diff) == np.max(diff):
        return "asymmetric"
    
    # Otherwise, it's uncertain
    else:
        return "unknown"