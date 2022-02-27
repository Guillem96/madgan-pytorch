from typing import Any, Callable, Iterator, Optional, Sequence
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import pandas as pd
import numpy as np


class WindowDataset(Dataset):
    """Dataset to iterate using sliding windows over a pandas DataFrame.

    Args:
        df (pd.DataFrame): Sorted DataFrame by time.
        window_size (int): Number of elements per window.
        window_slide (int): Step size between each window.

    """

    def __init__(self, df: pd.DataFrame, window_size: int,
                 window_slide: int) -> None:
        self.windows = _window_array(df.values, window_size, window_slide)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.as_tensor(self.windows[index].copy())

    def __len__(self) -> int:
        return self.windows.shape[0]


class LatentSpaceIterator(object):
    """Iterator that generates random sliding windows."""

    def __init__(self,
                 noise_shape: Sequence[int],
                 noise_type: str = "uniform") -> None:
        self.noise_fn: Callable[[Any], torch.Tensor]
        self.noise_shape = noise_shape
        if noise_type == "uniform":
            self.noise_fn = torch.rand
        elif noise_type == "normal":
            self.noise_fn = torch.randn
        else:
            raise ValueError(f"Unexpected noise type {noise_type}")

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def __next__(self) -> torch.Tensor:
        return self.noise_fn(*self.noise_shape)


def prepare_dataloader(ds: Dataset,
                       batch_size: int,
                       is_distributed: bool = False,
                       **kwargs: Any) -> DataLoader:
    """Creates a dataloader for training.

    Args:
        ds (Dataset): Training dataset.
        batch_size (int): DataLoader batch size.
        is_distributed (bool): Is the training distributed over multiple nodes? 
            Defaults to False.

    Returns:
        DataLoader: Data iterator ready to use.
    """
    sampler: Optional[DistributedSampler] = (DistributedSampler(ds)
                                             if is_distributed else None)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, **kwargs)


def _window_array(array: np.ndarray, window_size: int,
                  window_slide: int) -> np.ndarray:
    return np.vstack([
        array[i:i + window_size]
        for i in range(0, array.shape[0], window_slide)
    ])
