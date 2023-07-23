import sys
import os
import time

import pickle as pkl
import numpy as np
import h5py
from tqdm import tqdm


if __name__ == '__main__':
    np.random.seed(0)
    os.makedirs('sparse', exist_ok=True)
    os.makedirs('dense', exist_ok=True)

    num_examples = int(sys.argv[1])
    H = W = 420
    focal_plains = 7
    seq_len_range = (200, 600)

    write_sparse = bool(sys.argv[2])
    chunk_size = tuple(map(int, sys.argv[3:]))
    if not chunk_size:
        chunk_size = (1, focal_plains, H, W)

    sparse_writes = []
    dense_writes = []

    with h5py.File(f'dense/data.hdf5', 'w') as d:
        dataset = d.create_dataset('images',
                                   shape=(0, focal_plains, H, W),
                                   dtype=np.uint8,
                                   maxshape=(None, focal_plains, H, W),
                                   chunks=chunk_size,
                                   compression='gzip')

        offset = 0
        cum_seq_lens = []

        buffer = []

        for image_id in tqdm(range(num_examples)):
            seq_len = int(np.random.normal(loc=sum(seq_len_range)/2, scale=0.5))
            random_image_sequence = np.random.randint(0, 255, size=(seq_len, focal_plains, H, W), dtype=np.uint8)
            buffer.append(random_image_sequence)
            offset += seq_len
            cum_seq_lens.append(offset)

            if write_sparse:
                s = time.time()
                with h5py.File(f'sparse/{image_id}.hdf5', 'w') as f:
                    dset = f.create_dataset('image',
                                            data=random_image_sequence,
                                            chunks=chunk_size,
                                            compression='gzip')
                    dset[()] = random_image_sequence
                sparse_writes.append(time.time() - s)
            s = time.time()

            if (image_id + 1) % 10 == 0:
                dataset.resize(offset, axis=0)
                buffer = np.concatenate(buffer)
                dataset[-buffer.shape[0]:] = buffer
                buffer = []
                dense_writes.append(time.time() - s)

            if (image_id + 1) % 100 == 0:
                if sparse_writes:
                    print('sparse rolling mean [s]:', sum(sparse_writes[-100:]) / 100)
                print('dense rolling mean [s]:', dense_writes[-10] / 10)
        dataset.attrs['cum_seq_len'] = cum_seq_lens

    if sparse_writes:
        with open('sparse_times.pkl', 'wb') as f:
            pkl.dump(sparse_writes, f)

    with open('dense_times.pkl', 'wb') as f:
        pkl.dump(dense_writes, f)
