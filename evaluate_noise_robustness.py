"""Evaluate noise robustness of pretrained model.
"""

import os
import copy
import argparse

import torch
import pandas as pd
from braindecode.datautil.preprocess import preprocess, Preprocessor
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import pickle

from transforms import (
    AdditiveWhiteNoise, AutoRejectDrop, ensure_valid_positions)
from utils import load_data, split_dataset, parallel_preproc
from viz import plot_noise_robustness


def main(args):
    #%% 1- General stuff

    # Get the dirs of the pretrained models
    model_dirs = os.listdir(args.exp_dir)
    dir_names = [os.path.join(args.exp_dir, model_dir)
                 for model_dir in model_dirs]
    if len(dir_names) > 0:
        print(f'\nEvaluating noise robustness of {len(dir_names)} pretrained '
              f'models:\n{model_dirs}\n')
    else:
        raise FileNotFoundError(f'No models found under {args.exp_dir}.')

    # Make a directory for saving the results
    save_path = os.path.join(args.exp_dir, 'evaluate_noise_robustness')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # cuda parameters
    cuda = torch.cuda.is_available()  # check if GPU is available
    if cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = not args.deterministic
        torch.backends.cudnn.deterministic = args.deterministic
    else:
        pass  # torch.set_num_threads(args.n_jobs)

    #%% 2- Load, preprocess and window data

    windows_dataset = load_data(
        args.dataset, args.window_size_s, args.n_jobs)
    # Likely faster to reload entire data then split it as it should have been
    # cached by joblib at training time.

    _, _, test_set = split_dataset(
        windows_dataset, args.valid_size, args.test_size,
        random_state_valid=args.random_state_valid,
        random_state_test=args.random_state_test)

    #%% 3- Load pretrained models

    all_nets = {'autoreject': dict(), 'no_autoreject': dict()}
    for dir_name, model_dir in zip(dir_names, model_dirs):
        with open(os.path.join(dir_name, 'best_model.pkl'), 'rb') as f:
            if 'autoreject' in model_dir:
                all_nets['autoreject'][model_dir] = pickle.load(f)
            else:
                all_nets['no_autoreject'][model_dir] = pickle.load(f)

    # Define autoreject object
    ar = AutoRejectDrop(
        cv=5, random_state=args.seed, drop=False, n_jobs=args.n_jobs)

    #%% 4- Evaluate performance

    y_true_test = test_set.get_metadata()['target'].to_numpy()
    results = list()
    for denoising, nets in all_nets.items():
        if not nets:
            continue
        white_noise = AdditiveWhiteNoise(
            p=0.5, noise_strength=(0.5, 1), noise_std=(20, 50),
            recording_wise=True, random_state=args.seed)

        for noise_strength in [0, 0.25, 0.5, 0.75, 1]:
            print(f'Evaluating perf for noise_strength={noise_strength}...\n')
            test_set_copy = copy.deepcopy(test_set)

            # Corrupt data
            white_noise.noise_strength = noise_strength
            preprocess(test_set_copy, [Preprocessor(white_noise)])
            if denoising == 'autoreject':
                preproc_ar = [
                    Preprocessor(ensure_valid_positions, apply_on_array=False),
                    Preprocessor(ar, apply_on_array=False)
                ]
                test_set_copy = parallel_preproc(
                    test_set_copy, preproc_ar, args.n_jobs)

            # Get predictions and measure performance
            for model_dir, net in nets.items():
                y_pred_test = net.predict(test_set_copy)
                results.append({
                    'model_dir': model_dir,
                    'noise_strength': noise_strength,
                    'acc': accuracy_score(y_true_test, y_pred_test),
                    'bal_acc': balanced_accuracy_score(
                        y_true_test, y_pred_test),
                })

    results = pd.DataFrame(results)
    results[['dataset', 'model', 'dsf_type', 'denoising']] = \
        results['model_dir'].str.split('-', expand=True)
    results.to_csv(os.path.join(save_path, 'noise_robustness_results.csv'))

    #%% 5- Plot results

    fig, axes = plot_noise_robustness(results)
    fig.savefig(os.path.join(save_path, 'noise_robustness.png'))

    print(f'Results saved under {save_path}.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train models')

    # Plumbing
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='directory containing pretrained models')
    parser.add_argument('--seed', type=int, default=87,
                        help='random seed (default: 87)')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='number of parallel processes to use (default: 1)')
    parser.add_argument('--deterministic', type=bool, default=True,
                        help='make training deterministic (default: True)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of torch workers for data loading (default: 0')

    # Dataset
    parser.add_argument('--dataset', type=str, default='sleep_physionet',
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

    args = parser.parse_args()
    main(args)
