import sys
import os
import time

import pickle as pkl
import numpy as np
import h5py
from tqdm import tqdm


if __name__ == '__main__':
    np.random.seed(0)
    os.makedirs('binary', exist_ok=True)

    num_examples = int(sys.argv[1])
    H = W = 420
    focal_plains = 7
    seq_len_range = (200, 600)

    write_times = []

    with open(f'binary/data.dat', 'wb') as f:
        offset = 0
        cum_seq_lens = []

        for image_id in tqdm(range(num_examples)):
            seq_len = int(np.random.normal(loc=sum(seq_len_range)/2, scale=0.5))
            random_image_sequence = np.random.randint(0, 255, size=(seq_len, focal_plains, H, W), dtype=np.uint8)
            offset += seq_len
            cum_seq_lens.append(offset)

            f.write(random_image_sequence.reshape(-1).tobytes())  # shape: seq_len, focal_plains, H, W

    with open('binary/metadata.pkl', 'wb') as f:
        pkl.dump({
            'cum_seq_lens': cum_seq_lens,
            'example_shape': (focal_plains, H, W),
            'dtype': np.uint8
        }, f)

    with open('binary_write_times.pkl', 'wb') as f:
        pkl.dump(write_times, f)
