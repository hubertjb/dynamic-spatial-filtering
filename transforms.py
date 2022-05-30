"""Transforms.
"""

import copy

import mne
import numpy as np
from autoreject import AutoReject
from autoreject.autoreject import _check_data, _apply_interp, _apply_drop
from sklearn.utils import check_random_state


def ensure_valid_positions(epochs):
    """Make sure the EEG channel positions are valid.

    If channels are bipolar and referenced to M1 or M2, rename them to just the
    first derivation so that autoreject can be used.
    """
    ch_names = epochs.info['ch_names']
    if all(['-' not in c for c in ch_names]):  # Only monopolar channels
        pass
    elif all([c.endswith('-M1') or c.endswith('-M2') for c in ch_names]):
        ch_mapping = {c: c.split('-')[0] for c in ch_names}
        epochs.rename_channels(ch_mapping)
        epochs.set_montage('standard_1020')
    else:
        raise ValueError('Bipolar channels are referenced to another channel '
                         'than M1 or M2.')


class AutoRejectDrop(AutoReject):
    """Callable AutoReject with inplace processing and optional epoch dropping.

    See `autoreject.AutoReject`.
    """
    def __init__(self, drop=True, inplace=True, **kwargs):
        super().__init__(**kwargs)
        self.drop = drop
        self.inplace = inplace

    def __getstate__(self):
        """Necessary because the `AutoReject` object implements its own version.
        """
        state = super().__getstate__()
        for param in ['inplace', 'drop']:
            state[param] = getattr(self, param)
        return state

    def __setstate__(self, state):
        """Necessary because the `AutoReject` object implements its own version.
        """
        super().__setstate__(state)
        for param in ['inplace', 'drop']:
            setattr(self, param, state[param])

    def transform(self, epochs, return_log=False):
        """Same as AutoReject.transform(), but with inplace processing and
        optional epoch dropping.
        """
        # XXX : should be a check_fitted method
        if not hasattr(self, 'n_interpolate_'):
            raise ValueError('Please run autoreject.fit() method first')

        _check_data(epochs, picks=self.picks_, verbose=self.verbose)

        reject_log = self.get_reject_log(epochs)
        # First difference with the original code:
        epochs_clean = epochs if self.inplace else epochs.copy()
        _apply_interp(reject_log, epochs_clean, self.threshes_,
                      self.picks_, self.dots, self.verbose)

        if self.drop:  # Second difference with the original code
            _apply_drop(reject_log, epochs_clean, self.threshes_, self.picks_,
                        self.verbose)

        if return_log:
            return epochs_clean, reject_log
        else:
            return epochs_clean

    def __call__(self, epochs):
        epochs = self.fit_transform(epochs)


class AdditiveWhiteNoise(object):
    """Additive white noise.

    Parameters
    ----------
    p : float
        Probability that a channel receives white noise [0, 1].
    noise_strength : float | tuple
        Relative strength of the noise. The output of this transform is a
        convex combination of the original signal x and white noise n:

            y = (1 - w) * x + w * n

        If provided as a tuple (min_strength,max_strength), the relative
        strength will be uniformly sampled in the provided open interval.
    noise_std : float | tuple | None
        Standard deviation of the white noise. If provided as a tuple (min_std,
        max_std), the standard deviation will be uniformly sampled in the
        provided open interval. If None, the standard deviation will be the
        same as the standard deviation of the input signal.
    random_state : 'global' | np.random.RandomState | int | None
        Random state used to control noise parameters (channels to be
        corrupted, strength and standard deviation of noise). If 'global',
        random numbers will be generated with the `np.random` module so they
        use the global seed (this is useful to avoid duplicate augmentations
        when using transforms with num_workers > 1).
    noise_random_state : 'global' | np.random.RandomState | int | None
        Random state used to generate the white noise itself. If None, the
        random number generator will be initialized to the same as
        `random_state`'s. If 'global', random numbers will be generated with
        the `np.random` module so they use the global seed (this is useful to
        avoid duplicate augmentations when using transforms with
        num_workers > 1).

        NOTE: The two random states are kept separate so that it is possible to
              have identical recording-wise corruption on raw data and epoched
              data.

    recording_wise : bool
        If True and a 3D array (n_windows, n_channels, n_times) is passed to
        __call__, the same noise parameters will be used to generate the noise
        of all windows. If False, each window will be corupted with its own
        noise parameters.
    """
    __name__ = 'AdditiveWhiteNoise'

    def __init__(self, p, noise_strength, noise_std=None,
                 random_state='global', noise_random_state=None,
                 recording_wise=True):
        self.p = p
        self.noise_strength = noise_strength
        self.noise_std = noise_std
        self._set_random_states(random_state, noise_random_state)
        self.recording_wise = recording_wise

    def _set_random_states(self, random_state, noise_random_state):
        if random_state == 'global':
            self.rng = np.random
        else:
            self.rng = check_random_state(random_state)

        if noise_random_state == 'global':
            self.noise_rng = np.random
        elif noise_random_state is None:
            if random_state == 'global':
                self.noise_rng = np.random
            else:
                self.noise_rng = copy.deepcopy(self.rng)
        else:
            self.noise_rng = check_random_state(noise_random_state)

    def __call__(self, X, mask=None):
        """Generate and apply white noise to an mne.Epochs object.

        Parameters
        ----------
        X : np.ndarray | mne.Epochs
            Data to be corrupted.
        mask : np.ndarray | None
            If provided as a numpy array of bool with shape (n_channels,), will
            replace the mask that is normally sampled at every call.

        Returns
        -------
        np.ndarray :
            Corrupted data.
        """
        X_out = X._data if isinstance(X, mne.Epochs) else X

        if X_out.ndim == 2:
            n_channels, n_times = X_out.shape
        elif X_out.ndim == 3:
            if not self.recording_wise:
                all_Xi = [self.__call__(Xi, mask) for Xi in X_out]
                return np.stack(all_Xi, axis=0)
            n_windows, n_channels, n_times = X_out.shape
        else:
            raise ValueError(
                f'Data must have 2 or 3 dimensions, got {X_out.ndim}')

        # Pick channels
        if mask is None:
            mask = self.rng.binomial(1, self.p, n_channels) == 1
        n_bad_chs = sum(mask)

        if n_bad_chs > 0:
            if isinstance(self.noise_std, (int, float)):
                loc = 0
                scale = self.noise_std
            elif isinstance(self.noise_std, tuple):
                loc = np.zeros(n_bad_chs)
                scale = self.rng.uniform(
                    low=self.noise_std[0], high=self.noise_std[1],
                    size=n_bad_chs)
            elif self.noise_std is None:
                loc = np.zeros(n_bad_chs)
                scale = X[mask].std(axis=1)
            else:
                raise ValueError(
                    'noise_std must be an int, float, tuple or None, got '
                    f'{type(self.noise_std)}.')

            if X_out.ndim == 2:
                n = self.noise_rng.normal(
                    loc=loc, scale=scale, size=(n_times, n_bad_chs)).T
            elif X_out.ndim == 3:
                n = self.noise_rng.normal(
                    loc=loc, scale=scale, size=(n_windows, n_times, n_bad_chs))
                n = np.transpose(n, (0, 2, 1))

            if isinstance(self.noise_strength, tuple):
                w = self.rng.uniform(
                    low=self.noise_strength[0], high=self.noise_strength[1])
            else:
                w = self.noise_strength

            if X_out.ndim == 2:
                X_out[mask] = (1 - w) * X_out[mask] + w * n
            elif X_out.ndim == 3:
                X_out[:, mask] = (1 - w) * X_out[:, mask] + w * n

        return X_out


class Compose(object):
    """Compose several transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, X):
        for t in self.transforms:
            X = t(X)
        return X


def logm_cov(x):
    """Matrix logarithm using SVD.

    Dedicated linalg.logm function is slower than SVD-based approach for SPD
    matrices. See https://github.com/scipy/scipy/issues/12464

    Parameters
    ----------
    x : np.ndarray
        Window of shape (n_channels, n_times).

    Returns
    -------
    np.ndarray :
        Original window x.
    np.ndarray :
        Upper triangle of logm(x) of shape (n_channels * (n_channels + 1) / 2).
    """
    cov = np.cov(x, dtype=np.float32)

    U, S, V = np.linalg.svd(cov)
    logS = np.log(np.maximum(S, 1e-10))  # clamp the eigenvalues to avoid -inf
    out = U @ np.diag(logS) @ V

    triu_inds = np.triu_indices(x.shape[0])
    return x, out[triu_inds]
