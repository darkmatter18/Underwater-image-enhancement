from torch.utils.data import DataLoader


class CustomDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, torch_dataset, batch_size=1, num_threads=0, serial_batches=True, max_dataset_size=float("inf")):
        """
        Custom Dataloader Function

        :param torch_dataset: Torch dataset
        :param batch_size: no of examples per batch. Default: 1
        :param num_threads: number of workers for multiprocessing. Default: 0
        :param serial_batches: Whether the dataset has serial_batches or not. Default: True
        :param max_dataset_size: Max size of the dataset.
        """

        self.dataset = torch_dataset
        self.batch_size = batch_size
        self.max_dataset_size = max_dataset_size
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=not serial_batches,
            num_workers=int(num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.batch_size >= self.max_dataset_size:
                break
            yield data
