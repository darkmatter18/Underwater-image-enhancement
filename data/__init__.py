from .datasets import UWIEDataset
from .dataloaders import UWIEDataloader


def create_dataset(opt):
    """Create a Dataset using opt

    :param opt: The options
    :return: Loaded torch.tensor dataset
    """
    dataset = UWIEDataset(dataroot=opt.dataroot, phase=opt.phase, serial_batches=opt.serial_batches,
                          preprocess=opt.preprocess, flip=not opt.no_flip, load_size=opt.load_size,
                          crop_size=opt.crop_size)

    dataloader = UWIEDataloader(dataset, batch_size=opt.batch_size, num_threads=opt.num_threads,
                                serial_batches=opt.serial_batches, is_distributed=opt.is_distributed,
                                use_cuda=opt.use_cuda)

    return dataloader.load_data()
