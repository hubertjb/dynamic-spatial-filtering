"""Models.
"""

import torch
from torch import nn


def soft_thresholding(x, b, a=None):
    """Remap values between [-a, b] to 0, keep the rest linear.
    """
    if a is None:
        a = b
    return (torch.clamp(x - b, min=0) * (x > 0) +
            torch.clamp(x + a, max=0) * (x <= 0))


def logm_eig(A, spd=True):
    """Batched matrix logarithm through eigenvalue decomposition.

    Parameters
    ----------
    A : torch.Tensor
        Square matrices of shape (B, F, C, T).

    Returns
    -------
    torch.Tensor :
        Matrix logarithm of A.
    """
    e, v = torch.symeig(A, eigenvectors=True)
    e = torch.clamp(e, min=1e-10)  # clamp the eigenvalues to avoid -inf
    return v @ torch.diag_embed(
        torch.log(e), dim1=2, dim2=3) @ v.transpose(2, 3)


class SpatialFeatureExtractor(nn.Module):
    """Extract spatial features from input.
    """
    def __init__(self, kind, n_channels):
        super().__init__()
        self.kind = kind
        self.n_channels = n_channels
        self.inds = torch.triu_indices(n_channels, n_channels)

    @staticmethod
    def _cov(x):
        xm = x - x.mean(axis=3, keepdims=True)
        return xm @ xm.transpose(2, 3) / (x.shape[3] - 1)

    def forward(self, x):
        """
        x.shape = (B, F, C, T)
        """
        if self.kind == 'log_diag_cov':
            out = torch.log(torch.var(x, 3, unbiased=True))
            out[torch.isneginf(out)] = 0
        elif self.kind == 'logm_cov_eig':
            cov = self._cov(x)
            logm_cov = logm_eig(cov)
            out = logm_cov[:, :, self.inds[0], self.inds[1]]
        else:
            out = None

        return out

    @property
    def n_outputs(self):
        if self.kind == 'log_diag_cov':
            return self.n_channels
        else:
            return int(self.n_channels * (self.n_channels + 1) / 2)


class DynamicSpatialFilter(nn.Module):
    """Dynamic spatial filter module.

    Input: (B, F, C, T) [F is the number of filters]
    Output: (B, F, C', T) [transformed input]

    Parameters
    ----------
    n_channels : int
        Number of input channel.
    mlp_input : str
        What to feed the MLP. See SpatialFeatureExtractor.
    n_hidden : int | None
        Number of hidden neurons in the MLP. If None, use `ratio`.
    ratio : float
        If `n_hidden` is None, the number of hidden neurons in the MLP is
        computed as int(ratio * n_inputs).
    n_out_channels : int | None
        Number of output ("virtual") channels in the DSF-based models (only
        affects DSF models). If None, n_out_channels = n_channels.
    apply_soft_thresholding : bool
        If True, apply soft thresholding to the spatial filter matrix W.
    return_att : bool
        If True, `forward()` returns attention values as well. Used for
        inspecting the model.
    """
    def __init__(self, n_channels, mlp_input='log_diag_cov', n_hidden=None,
                 ratio=1, n_out_channels=None, apply_soft_thresh=False,
                 return_att=False):
        super().__init__()
        self.apply_soft_thresh = apply_soft_thresh
        self.return_att = return_att

        # Initialize spatial feature extractor
        self.feat_extractor = SpatialFeatureExtractor(
            mlp_input, n_channels)
        n_inputs = self.feat_extractor.n_outputs
        if n_hidden is None:
            n_hidden = int(ratio * n_inputs)

        # Define number of outputs
        if n_out_channels is None:
            n_out_channels = n_channels
        self.n_out_channels = n_out_channels
        n_outputs = n_out_channels * (n_channels + 1)

        self.mlp = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs)
        )

    def forward(self, x):
        if isinstance(x, list):  # logm was computed on CPU with transforms
            x, feats = x
            feats = feats.unsqueeze(1)
        else:
            feats = None
        if x.ndim == 3:
            b, c, _ = x.shape
            f = 1
            x = x.unsqueeze(1)
        elif x.ndim == 4:
            b, f, c, _ = x.shape

        mlp_out = self.mlp(self.feat_extractor(x) if feats is None else feats)

        W = mlp_out[:, :, self.n_out_channels:].view(
            b, f, self.n_out_channels, c)
        if self.apply_soft_thresh:
            W = soft_thresholding(W, 0.1)
        bias = mlp_out[:, :, :self.n_out_channels].view(
            b, f, self.n_out_channels, 1)
        out = W @ x + bias

        if self.return_att:
            return out, (W, bias)
        else:
            return out
