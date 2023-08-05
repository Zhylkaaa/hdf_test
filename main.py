import time
import json
from collections import defaultdict

from tqdm import tqdm

from datamodules import get_dense_dataloader, get_sparse_dataloader, get_binary_dataloader

dataloaders_map = {
    'sparse': get_sparse_dataloader,
    'dense': get_dense_dataloader,
    'binary': get_binary_dataloader
}


def test_dataloader(data_path, num_workers, batch_size):
    dataloader = dataloaders_map[data_path](data_path, batch_size=batch_size, num_workers=num_workers)

    for batch in tqdm(dataloader):
        pass
    return batch


if __name__ == '__main__':
    test_results = {}

    for num_workers in [0, 2]:
        for batch_size in [2, 32]:
            for data_path in ['dense', 'binary']:
                times = []
                for i in range(3):
                    s = time.time()
                    batch = test_dataloader(data_path, num_workers, batch_size)
                    e = time.time()
                    times.append(e - s)
                print(batch.shape, batch.dtype)
                test_results[f'{data_path}_{num_workers}_{batch_size}'] = sum(times) / len(times)
                print(sum(times) / len(times))

    with open('speed_test_datasets.json', 'w') as f:
        json.dump(test_results, f)
