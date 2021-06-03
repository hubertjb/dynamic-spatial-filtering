"""Training of neural network.
"""

import os
import argparse

import mne
import torch
from torch import nn
from braindecode import EEGClassifier
from braindecode.models import SleepStagerChambon2018
from braindecode.util import set_random_seeds
from skorch.helper import predefined_split
from skorch.callbacks import (
    Checkpoint, EarlyStopping, EpochScoring, LRScheduler)
from sklearn.metrics import (
    confusion_matrix, classification_report, balanced_accuracy_score)
from sklearn.utils.class_weight import compute_class_weight
import pickle

from transforms import Compose, AdditiveWhiteNoise, logm_cov
from models import DynamicSpatialFilter
from utils import (
    load_data, apply_autoreject, split_dataset, none_or_int, get_exp_name,
    seed_np_rng)


mne.set_log_level('WARNING')
torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):

    #%% 1- General stuff
    cuda = torch.cuda.is_available()  # check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = not args.deterministic
        torch.backends.cudnn.deterministic = args.deterministic
    else:
        pass  # torch.set_num_threads(args.n_jobs)

    # Create savedir
    dir_name = get_exp_name(args.dataset, args.model, args.dsf_type,
                            args.denoising)
    save_path = os.path.join(args.save_dir, dir_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #%% 2- Load, preprocess and window data
    windows_dataset = load_data(args.dataset, args.window_size_s, args.n_jobs)

    if args.denoising == 'autoreject':
        windows_dataset = apply_autoreject(
            windows_dataset, args.seed, args.n_jobs)

    # Split into train, valid and test sets
    available_classes = windows_dataset.get_metadata()['target'].unique()
    train_set, valid_set, test_set = split_dataset(
        windows_dataset, args.valid_size, args.test_size,
        random_state_valid=args.random_state_valid,
        random_state_test=args.random_state_test)
    del windows_dataset

    if args.denoising == 'data_augm':
        train_set.transform = AdditiveWhiteNoise(
            p=0.5, noise_strength=(0.5, 1), noise_std=(20, 50),
            recording_wise=False)

    # Extract weights to balance the loss function
    y_true_train = train_set.get_metadata()['target'].to_numpy()
    train_weights = torch.Tensor(compute_class_weight(
        'balanced', classes=available_classes, y=y_true_train)).to(device)

    #%% 3- Create model

    # Set random seed to be able to reproduce results
    set_random_seeds(seed=args.seed, cuda=cuda)

    # Extract number of channels and time steps from dataset
    n_classes = len(available_classes)
    n_channels = train_set[0][0].shape[0]
    if args.dsf_type != 'vanilla':
        if args.dsf_type == 'dsfd':
            mlp_input = 'log_diag_cov'
            dsf_soft_thresh = False
        elif args.dsf_type == 'dsfm_st':
            mlp_input = 'logm_cov_eig'
            dsf_soft_thresh = True

            # Use CPU to compute logm, it's faster than pytorch with cuda
            train_set.transform = logm_cov if train_set.transform[0] is None \
                else Compose([train_set.transform[0], logm_cov])
            valid_set.transform = logm_cov
            test_set.transform = logm_cov

        else:
            raise ValueError(
                f'dsf_type must be None, dsfd or dsfm_st, got {args.dsf_type}')
        dsf = DynamicSpatialFilter(
            n_channels, mlp_input=mlp_input,
            n_out_channels=args.dsf_n_out_channels,
            apply_soft_thresh=dsf_soft_thresh)
        n_channels = dsf.n_out_channels

    input_size_samples = len(train_set.datasets[0].windows.times)

    sfreq = train_set.datasets[0].windows.info['sfreq']
    if args.model == 'stager_net':
        model = SleepStagerChambon2018(
            n_channels, sfreq, n_conv_chs=args.n_conv_chs,
            input_size_s=input_size_samples / sfreq, pad_size_s=0.1,
            n_classes=n_classes, dropout=args.dropout, apply_batch_norm=True
            ).to(device)
    else:
        raise NotImplementedError

    if args.dsf_type != 'vanilla':
        model = nn.Sequential(dsf, model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nModel has {n_params} trainable parameters.\n')

    if torch.cuda.device_count() > 1:  # Parallelize model over GPUs
        print(f'\nUsing {torch.cuda.device_count()} GPUs.\n')
        model = nn.DataParallel(model)

    #%% 4- Train and evaluate model

    cp = Checkpoint(dirname=save_path)
    early_stopping = EarlyStopping(patience=args.patience)
    train_bal_acc = EpochScoring(
        scoring='balanced_accuracy', on_train=True, name='train_bal_acc',
        lower_is_better=False)
    valid_bal_acc = EpochScoring(
        scoring='balanced_accuracy', on_train=False, name='valid_bal_acc',
        lower_is_better=False)

    callbacks = [
        ('cp', cp),
        ('patience', early_stopping),
        ('train_bal_acc', train_bal_acc),
        ('valid_bal_acc', valid_bal_acc),
    ]

    if args.cosine_annealing:
        callbacks.append(('lr_scheduler', LRScheduler(
            'CosineAnnealingLR', T_max=args.n_epochs - 1)))

    net = EEGClassifier(
        module=model,
        criterion=torch.nn.CrossEntropyLoss,
        criterion__weight=train_weights,
        optimizer=torch.optim.AdamW,
        optimizer__weight_decay=args.weight_decay,
        train_split=predefined_split(valid_set),
        optimizer__lr=args.lr,
        max_epochs=args.n_epochs,
        batch_size=args.batch_size,
        iterator_train__shuffle=True,
        iterator_train__num_workers=args.num_workers,
        iterator_valid__num_workers=args.num_workers,
        iterator_train__worker_init_fn=seed_np_rng,
        callbacks=callbacks,
        device=device
    )
    net.fit(train_set, y=None)

    # Load best model
    net.initialize()
    net.load_params(checkpoint=cp)

    # Pickle best model
    with open(os.path.join(save_path, 'best_model.pkl'), 'wb') as f:
        net.train_split = None  # Avoid pickling the validation set
        pickle.dump(net, f)

    #%% 5- Evaluate performance

    y_true_test = test_set.get_metadata()['target'].to_numpy()
    y_pred_test = net.predict(test_set)
    test_bal_acc = balanced_accuracy_score(y_true_test, y_pred_test) * 100

    print('\nTest results:\n-------------\n')
    print(f'Balanced accuracy: {test_bal_acc:0.2f}%\n')
    print('Confusion matrix:')
    print(confusion_matrix(y_true_test, y_pred_test))
    print('\nClassification report:')
    print(classification_report(y_true_test, y_pred_test))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train models')

    # Plumbing
    parser.add_argument('--save_dir', type=str, default='./runs',
                        help='save results in this directory (default: ./runs) ')
    parser.add_argument('--seed', type=int, default=87,
                        help='random seed (default: 87)')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='number of parallel processes to use (default: 1)')
    parser.add_argument('--deterministic', type=bool, default=True,
                        help='make training deterministic (default: True)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of torch workers for data loading (default: 0')

    # Dataset
    parser.add_argument('--dataset', type=str, default='pc18_debug',
                        choices=['sleep_physionet', 'pc18', 'pc18_debug'],
                        help='sleep_physionet|pc18|pc18_debug')
    parser.add_argument('--valid_size', type=float, default=0.2,
                        help='proportion of dataset to keep for validation (default: 0.2)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='proportion of dataset to keep for testing (default: 0.2)')
    parser.add_argument('--random_state_valid', type=int, default=87,
                        help='random state for splitting valid set (default: 87)')
    parser.add_argument('--random_state_test', type=int, default=87,
                        help='random state for splitting test set (default: 87)')

    # Preprocessing
    parser.add_argument('--window_size_s', type=int, default=30,
                        help='size of input windows in seconds (default: 30)')

    # Model hyperparameters
    parser.add_argument('--model', type=str, default='stager_net',
                        choices=['stager_net'],
                        help='model name (default: stager_net)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout for fully connected layer (default: 0.5)')
    parser.add_argument('--n_conv_chs', type=int, default=16,
                        help='number of convolutional channels (default: 16)')
    parser.add_argument('--dsf_type', type=str, default='vanilla',
                        choices=['vanilla', 'dsfd', 'dsfm_st'],
                        help='Type of DSF module (default: None)')
    parser.add_argument('--dsf_n_out_channels', type=none_or_int, default=None,
                        help='number of DSF virtual channels (default: None)')
    parser.add_argument('--denoising', type=str, default='no_denoising',
                        choices=['no_denoising', 'autoreject', 'data_augm'],
                        help='no_denoising|autoreject|data_augm')

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size (default: 64)')
    parser.add_argument('--n_epochs', type=int, default=5,
                        help='number of training epochs (default: 5)')
    parser.add_argument('--patience', type=int, default=5,
                        help='patience for training epochs (default: 5)')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='weight decay (default: 0.001)')
    parser.add_argument('--cosine_annealing', type=bool, default=True,
                        help='whether to use cosine annealing (default: True)')

    args = parser.parse_args()
    main(args)
