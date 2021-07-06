from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class UWIEDataloader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, torch_dataset: Dataset, batch_size: int = 1, is_distributed: bool = False,
                 use_cuda: bool = False, is_test: bool = True):
        """
        Custom Dataloader Function

        :param torch_dataset: Torch dataset
        :param batch_size: no of examples per batch. Default: 1
        :param is_distributed: Whether distributed training is using or not. Default: False
        :param use_cuda: Whether using cuda training is using or not. Default: False
        """

        self.is_distributed = is_distributed
        self.dataset = torch_dataset
        self.batch_size = batch_size

        # Variables for distributed training
        train_sampler = DistributedSampler(self.dataset) if self.is_distributed else None
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None or is_test,
            sampler=train_sampler,
            **kwargs)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data
