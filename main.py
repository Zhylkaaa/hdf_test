import sys
import time
import json

from tqdm import tqdm

from datamodules import get_dense_dataloader, get_sparse_dataloader


def test_dataloader(data_path, num_workers, batch_size, num_samples):
    if data_path == 'sparse':
        dataloader = get_sparse_dataloader(data_path, batch_size=batch_size, num_workers=num_workers, num_samples=num_samples)
    else:
        dataloader = get_dense_dataloader(data_path, batch_size=batch_size, num_workers=num_workers, num_samples=num_samples)

    for batch in dataloader:
        pass
    return batch


if __name__ == '__main__':
    if len(sys.argv) > 1:
        num_samples = int(sys.argv[1])
    else:
        num_samples = 20

    test_results = {}

    for num_workers in [0]:

        for batch_size in [1, 2]:
            for data_path in ['dense']:
                times = []
                for i in range(3):
                    s = time.time()
                    batch = test_dataloader(data_path, num_workers, batch_size, num_samples=num_samples)
                    e = time.time()
                    times.append(e - s)
                print(times)
                test_results[f'{data_path}_{num_workers}_{batch_size}'] = sum(times) / len(times)
                print(sum(times) / len(times))

    with open('speed_test_datasets.json', 'w') as f:
        json.dump(test_results, f)
