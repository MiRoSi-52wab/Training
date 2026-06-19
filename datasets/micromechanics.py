"""
PyTorch Dataset for LS-KAN-FNO HDF5 files (single entry point for all models,
per PROJECT_CONTEXT.md design principle 1).

Reads files written by generation/generate_dataset.py and returns Voigt-notation
float32 tensors. Voigt -> Mandel conversion happens inside the model
(models/ls_fno.py::_to_mandel_state), not here.

Usage:
    from datasets.micromechanics import DataLoaderFactory
    loaders = DataLoaderFactory.from_config(config['data'])
    train_loader = loaders['train']
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MicromechanicsDataset(Dataset):
    """
    Lazy, per-sample reader for one LS-KAN-FNO HDF5 dataset file.

    Uses the train_idx/val_idx/test_idx arrays already stored in the file's
    metadata group attrs (written by generation/generate_dataset.py) — samples
    are never re-split here.

    Args:
        h5_path: path to a dataset file.
        split:   'train', 'val', 'test', or 'all'.
    """

    def __init__(self, h5_path, split: str = "all"):
        self.h5_path = Path(h5_path)
        self._file = None  # opened lazily; h5py file handles aren't fork-safe

        with h5py.File(self.h5_path, "r") as f:
            meta = f["metadata"].attrs
            if split == "all":
                self.indices = np.arange(int(meta["n_samples"]))
            else:
                key = f"{split}_idx"
                if key not in meta:
                    raise ValueError(f"No '{key}' in {self.h5_path}'s metadata attrs.")
                self.indices = np.asarray(meta[key])

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        self._ensure_open()
        idx = int(self.indices[i])
        f = self._file
        return {
            "C_field":    torch.from_numpy(f["C_field"][idx]).float(),
            "eps_bar":    torch.from_numpy(f["eps_bar"][idx]).float(),
            "eps_star":   torch.from_numpy(f["eps_star"][idx]).float(),
            "tau_star":   torch.from_numpy(f["tau_star"][idx]).float(),
            "sigma_star": torch.from_numpy(f["sigma_star"][idx]).float(),
            "n_iter":     int(f["n_iter"][idx]),
            "converged":  bool(f["converged"][idx]),
        }

    def __getstate__(self):
        # Drop the open file handle before pickling (DataLoader workers with
        # start_method='spawn'); each worker reopens lazily on first access.
        state = self.__dict__.copy()
        state["_file"] = None
        return state


class DataLoaderFactory:
    """Builds train/val/test DataLoaders from a config dict."""

    @staticmethod
    def from_config(cfg: dict) -> dict:
        """
        Args:
            cfg: dict with keys:
                train_path:  path to the HDF5 file (required).
                val_path:    defaults to train_path (same file, val_idx split).
                test_path:   defaults to train_path (same file, test_idx split).
                batch_size:  default 16.
                num_workers: default 0 (safe default for local CPU runs).
                pin_memory:  default True iff CUDA is available.

        Returns:
            {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
        """
        train_path = cfg["train_path"]
        val_path = cfg.get("val_path", train_path)
        test_path = cfg.get("test_path", train_path)
        batch_size = int(cfg.get("batch_size", 16))
        num_workers = int(cfg.get("num_workers", 0))
        pin_memory = bool(cfg.get("pin_memory", torch.cuda.is_available()))

        train_ds = MicromechanicsDataset(train_path, split="train")
        val_ds = MicromechanicsDataset(val_path, split="val")
        test_ds = MicromechanicsDataset(test_path, split="test")

        common = dict(num_workers=num_workers, pin_memory=pin_memory)
        return {
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common),
            "val":   DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common),
            "test":  DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common),
        }
