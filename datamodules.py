import os
import torch
from torch.utils.data import Dataset, DataLoader
from multiprocessing import get_context
import h5py
import numpy as np
import pickle as pkl
import mmap
from concurrent.futures import ThreadPoolExecutor


class SparseDataset(Dataset):
    def __init__(self, data_path, num_samples, focal_start=2, focal_end=5):
        super().__init__()
        self.data_path = data_path
        self.num_samples = num_samples
        self.num_files = len([f for f in os.listdir(self.data_path) if not f.startswith('.')])
        self.focal_start = focal_start
        self.focal_end = focal_end

    def __len__(self):
        return self.num_files

    def __getitem__(self, index):
        with h5py.File(os.path.join(self.data_path, str(index)+'.hdf5')) as f:
            images = f['image']
            sampled_idxs = np.arange(0, images.shape[0], dtype=np.int64)
            sampled_idxs = np.random.choice(sampled_idxs, replace=False, size=self.num_samples)
            sampled_idxs.sort()
            return np.stack([images[idx, self.focal_start:self.focal_end] for idx in sampled_idxs])


class DenseDataset(Dataset):
    def __init__(self, data_path, num_samples, focal_start=2, focal_end=5):
        super().__init__()
        self.data_path = data_path
        self.num_samples = num_samples
        self.data = h5py.File(os.path.join(data_path, 'data.hdf5'))
        self.images = self.data['images']
        self.cum_seq_lens = self.images.attrs['cum_seq_len']
        self.focal_start = focal_start
        self.focal_end = focal_end

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries (e.g. file descriptors).
        del state['data']
        del state['images']
        del state['cum_seq_lens']
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        # Restore hdf5 dataset.
        self.data = h5py.File(os.path.join(self.data_path, 'data.hdf5'))
        self.images = self.data['images']
        self.cum_seq_lens = self.images.attrs['cum_seq_len']

    def __len__(self):
        return len(self.cum_seq_lens)

    def __getitem__(self, index):
        offset = self.cum_seq_lens[index - 1] if index else 0
        #seq_len = self.cum_seq_lens[index] - offset
        sampled_idxs = np.arange(offset, self.cum_seq_lens[index], dtype=np.int64)
        sampled_idxs = np.random.choice(sampled_idxs,  replace=False, size=self.num_samples)
        sampled_idxs.sort()
        return np.stack([self.images[idx, self.focal_start:self.focal_end] for idx in sampled_idxs])


class BinaryDataset(Dataset):
    def __init__(self, data_path, num_samples, focal_start=2, focal_end=5):
        super().__init__()
        self.data_path = data_path
        self.num_samples = num_samples
        self.focal_start = focal_start
        self.focal_end = focal_end

        self._load_memmap()

    def _load_memmap(self):
        self.f = open(os.path.join(self.data_path, 'data.dat'), 'rb')
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.mm.madvise(mmap.MADV_WILLNEED)
        # try:
        #     self.mm.madvise(mmap.MADV_HUGEPAGE)
        # except Exception as e:
        #     print('Failed to initialize HUGEPAGE')
        #     print(e)

        with open(os.path.join(self.data_path, 'metadata.pkl'), 'rb') as f:
            metadata = pkl.load(f)

        self.cum_seq_lens = metadata['cum_seq_lens']
        self.example_shape = metadata['example_shape']
        self.image_size = np.prod(self.example_shape[-2:])
        self.example_size = np.prod(self.example_shape)
        self.focal_offset_start = self.focal_start * self.image_size
        self.focal_offset_end = self.focal_end * self.image_size
        self.images = np.frombuffer(self.mm, dtype=metadata['dtype'])
        self.executor = ThreadPoolExecutor(max_workers=self.num_samples)

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries (e.g. file descriptors).
        del state['mm']
        del state['images']
        del state['f']
        del state['executor']
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        # Restore hdf5 dataset.
        self._load_memmap()

    def __len__(self):
        return len(self.cum_seq_lens)

    def _select_id(self, idx):
        return self.images[idx * self.example_size + self.focal_offset_start:
                           idx * self.example_size + self.focal_offset_end]

    # TODO: verify correctness
    def __getitem__(self, index):
        offset = self.cum_seq_lens[index - 1] if index else 0
        #seq_len = self.cum_seq_lens[index] - offset
        sampled_idxs = np.arange(offset, self.cum_seq_lens[index], dtype=np.int64)
        sampled_idxs = np.random.choice(sampled_idxs,  replace=False, size=self.num_samples)
        sampled_idxs.sort()
        #with ThreadPoolExecutor(max_workers=self.num_samples) as executor:
        return np.stack(
            list(self.executor.map(self._select_id, sampled_idxs))
        ).reshape(self.num_samples,
                  self.focal_end - self.focal_start,
                  *self.example_shape[-2:])
            # return np.stack([self.images[idx * self.example_size + self.focal_offset_start:
            #                              idx * self.example_size + self.focal_offset_end]
            #                  for idx in sampled_idxs]).reshape(self.num_samples,
            #                                                    self.focal_end - self.focal_start,
            #                                                    *self.example_shape[-2:])


def get_sparse_dataloader(data_path, num_samples=20, batch_size=1, num_workers=0, context='spawn', shuffle=True):
    return DataLoader(
        SparseDataset(data_path, num_samples),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        multiprocessing_context=get_context(context) if num_workers else None
    )


def get_dense_dataloader(data_path, num_samples=20, batch_size=1, num_workers=0, context='spawn', shuffle=True):
    return DataLoader(
        DenseDataset(data_path, num_samples),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        multiprocessing_context=get_context(context) if num_workers else None
    )


def get_binary_dataloader(data_path, num_samples=20, batch_size=1, num_workers=0, context='spawn', shuffle=True):
    return DataLoader(
        BinaryDataset(data_path, num_samples),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        multiprocessing_context=get_context(context) if num_workers else None
    )
