"""Utility functions.
"""

import os
import copy
from functools import partial

import torch
import numpy as np
from joblib import Memory, Parallel, delayed
from sklearn.model_selection import train_test_split
from braindecode.preprocessing.preprocess import (
    preprocess, Preprocessor, _preprocess)
from braindecode.preprocessing.windowers import _create_windows_from_events
from braindecode.datasets import BaseConcatDataset
from braindecode.datasets.sleep_physionet import SleepPhysionet

from transforms import ensure_valid_positions, AutoRejectDrop
from datasets import PC18


# Cache data to speed up experiments
joblib_cache_dir = os.environ.get('JOBLIB_CACHE_DIR', './')
if joblib_cache_dir.lower() == 'none':
    joblib_cache_dir = None  # Don't cache data
memory = Memory(joblib_cache_dir, verbose=50)


def scale(x, k):
    return k * x


def cast(x, dtype):
    return x.astype(dtype)


@memory.cache(ignore=['n_jobs'])
def load_data(dataset, window_size_s, n_jobs):
    """Load, preprocess and window data.
    """
    if dataset == 'sleep_physionet':
        dataset = SleepPhysionet(
            subject_ids=[0, 1, 2], recording_ids=[1], crop_wake_mins=30)
        mapping = {  # Merge stages 3 and 4 following AASM standards
            'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3': 3,
            'Sleep stage 4': 3,
            'Sleep stage R': 4
        }
        preproc = [
            Preprocessor('filter', l_freq=None, h_freq=30, n_jobs=n_jobs),
            Preprocessor(scale, k=1e6),
            Preprocessor(cast, dtype=np.float32)
        ]
        preprocess(dataset, preproc)

    elif dataset.startswith('pc18'):
        if dataset == 'pc18_debug':
            subject_ids = [989, 990, 991]
        elif dataset == 'pc18_hundred_files':
            subject_ids = range(989, 1089)
        else:
            subject_ids = 'training'

        ch_names = ['F3-M2', 'F4-M1', 'O1-M2', 'O2-M1']
        preproc = [
            Preprocessor('pick_channels', ch_names=ch_names, ordered=True),
            Preprocessor('filter', l_freq=None, h_freq=30, n_jobs=1),
            Preprocessor('resample', sfreq=100., n_jobs=1),
            Preprocessor(scale, k=1e6),
            Preprocessor(cast, dtype=np.float32)
        ]

        window_size_samples = int(window_size_s * 100)
        mapping = {'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4}
        windower = partial(
            _create_windows_from_events, infer_mapping=False,
            infer_window_size_stride=False, trial_start_offset_samples=0,
            trial_stop_offset_samples=0,
            window_size_samples=window_size_samples,
            window_stride_samples=window_size_samples, mapping=mapping)

        dataset = PC18(subject_ids=subject_ids, preproc=preproc,
                       windower=windower, n_jobs=n_jobs)

    else:
        raise NotImplementedError

    return dataset


def parallel_preproc(windows_dataset, preproc, n_jobs):
    """Apply preprocessor in parallel on BaseDatasets.
    """
    def _apply_preproc(ds, preproc):
        if ds.windows.preload:  # Deep copy so that numpy arrays are modifiable
            ds = copy.deepcopy(ds)
        try:
            _preprocess(ds, None, preproc)
        except Exception as e:
            print(e)
            print('Not applying preproc')
        return ds

    preproc_ds = Parallel(n_jobs=n_jobs)(delayed(_apply_preproc)(
        windows_dataset.datasets.pop(0), preproc)
        for _ in range(len(windows_dataset.datasets)))

    return BaseConcatDataset(preproc_ds)


@memory.cache(ignore=['n_jobs'])
def apply_autoreject(windows_dataset, random_state, n_jobs):
    ar = AutoRejectDrop(cv=5, random_state=random_state, drop=False, n_jobs=1)
    preproc = [
        Preprocessor(ensure_valid_positions, apply_on_array=False),
        Preprocessor(ar, apply_on_array=False)
    ]
    return parallel_preproc(windows_dataset, preproc, n_jobs)


def split_dataset(base_concat_ds, valid_size, test_size,
                  random_state_valid=None, random_state_test=None):
    """Split dataset into train, valid and test sets.

    Parameters
    ----------
    base_concat_ds : braindecode.datasets.BaseConcatDataset
        Dataset to split.
    valid_size : float
        Proportion of the dataset to include in the valid split.
    test_size : float
        Proportion of the dataset to include in the test split.
    random_state_valid : int | np.random.RandomState | None
        Controls the shuffling applied to the data before applying the
        validation split.
    random_state_test : int | np.random.RandomState | None
        Controls the shuffling applied to the data before applying the test
        split.

    Returns
    -------
    BaseConcatDataset, BaseConcatDataset, BaseConcatDataset :
        Train, valid and test splits.
    """
    rec_inds = np.arange(len(base_concat_ds.datasets))

    train_valid_inds, test_inds = train_test_split(
        rec_inds, test_size=test_size, random_state=random_state_test)
    train_inds, valid_inds = train_test_split(
        train_valid_inds, test_size=valid_size / (1 - test_size),
        random_state=random_state_valid)

    split_ds = base_concat_ds.split(
        [train_inds.tolist(), valid_inds.tolist(), test_inds.tolist()])

    return split_ds['0'], split_ds['1'], split_ds['2']


def none_or_int(value, value_type=str):
    if value == 'None':
        return None
    return int(value)


def get_exp_name(dataset, model, dsf_type, denoising):
    return f'{dataset}-{model}-{dsf_type}-{denoising}'


def seed_np_rng(worker_id):
    """Seed numpy random number generator for DataLoader with num_workers > 1.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
