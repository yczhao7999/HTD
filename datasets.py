import numpy as np
from scipy import io
from torch.utils.data import Dataset

DATA_KEYS = ["data", "xuzhou", "sandiego", "cri", "segundo", "aviris2", "avi"]
GT_KEYS = [
    "gt",
    "groundtruth",
    "ground_truth",
    "truth",
    "mask",
    "label",
    "labels",
    "xuzhou_gt",
]
PRIOR_KEYS = [
    "tgt",
    "target",
    "prior",
    "target_spectrum",
    "tgt_mean",
    "tgt_robust",
    "mean",
    "robust",
]


def _mat(path, keys):
    v = {
        k: x
        for k, x in io.loadmat(path).items()
        if not k.startswith("__") and isinstance(x, np.ndarray)
    }
    for k in keys:
        if k in v:
            return v[k]
    n = [x for x in v.values() if np.issubdtype(x.dtype, np.number)]
    if len(n) == 1:
        return n[0]
    raise KeyError(path)


def resample_spectra(x, bands=256):
    x = np.asarray(x, dtype=np.float32)
    old = np.linspace(0, 1, x.shape[-1])
    new = np.linspace(0, 1, bands)
    flat = x.reshape(-1, x.shape[-1])
    return (
        np.stack([np.interp(new, old, r) for r in flat])
        .reshape(*x.shape[:-1], bands)
        .astype(np.float32)
    )


def prepare_spectra(x, bands=None):
    """Keep native spectral length unless an explicit target length is given."""
    x = np.asarray(x, dtype=np.float32)
    return x if bands is None else resample_spectra(x, bands)


def scene_array(path, gt_path=None):
    c = np.asarray(_mat(path, DATA_KEYS), np.float32)
    g = None if gt_path is None else np.asarray(_mat(gt_path, GT_KEYS)).squeeze()
    if g is not None and c.shape[:2] != g.shape and c.shape[1:] == g.shape:
        c = np.transpose(c, (1, 2, 0))
    if c.ndim != 3 or (g is not None and c.shape[:2] != g.shape):
        raise ValueError((c.shape, None if g is None else g.shape))
    return c, None if g is None else (g > 0).astype(np.int64)


def normalize_scene(x, stats=None, return_stats=False):
    x = np.asarray(x, dtype=np.float32)
    if stats is None:
        minimum = float(np.nanmin(x))
        maximum = float(np.nanmax(x))
    else:
        minimum, maximum = (float(value) for value in stats)
    normalized = ((x - minimum) / (maximum - minimum + 1e-8)).astype(np.float32)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    normalized = np.clip(normalized, 0.0, 1.0)
    if return_stats:
        return normalized, (minimum, maximum)
    return normalized


def load_prior(path, bands=None, normalization_stats=None):
    prior = np.asarray(_mat(path, PRIOR_KEYS), np.float32).squeeze()
    prior = prior[None, :] if prior.ndim == 1 else prior
    if prior.ndim == 2 and prior.shape[0] > prior.shape[1]:
        prior = prior.T
    if normalization_stats is not None:
        prior = normalize_scene(prior, stats=normalization_stats)
    return prepare_spectra(prior, bands)


class SpectralDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def load_xuzhou(
    data_path="data/xuzhou/xuzhou.mat",
    gt_path="data/xuzhou/xuzhou_gt.mat",
    bands=None,
):
    c = np.asarray(_mat(data_path, DATA_KEYS), np.float32)
    y = np.asarray(_mat(gt_path, GT_KEYS)).squeeze()
    if c.shape[:2] != y.shape and c.shape[1:] == y.shape:
        c = np.transpose(c, (1, 2, 0))
    c = normalize_scene(c)
    c = prepare_spectra(c, bands)
    output_bands = c.shape[-1]
    x = c.reshape(-1, output_bands)
    y = y.reshape(-1).astype(np.int64)
    m = y > 0
    return x[m], y[m] - 1
