from .datasets import CustomDataset
from .dataloaders import CustomDatasetDataLoader


def create_dataset(opt):
    dataset = CustomDataset(dataroot=opt.dataroot, phase=opt.phase, max_dataset_size=opt.max_dataset_size,
                            direction=opt.direction, input_nc=opt.input_nc, output_nc=opt.output_nc,
                            serial_batches=opt.serial_batches, preprocess=opt.preprocess, flip=opt.flip,
                            load_size=opt.load_size, crop_size=opt.crop_size)
    dataloader = CustomDatasetDataLoader(dataset, batch_size=opt.batch_size, num_threads=opt.num_threads,
                                         serial_batches=opt.serial_batches, max_dataset_size=opt.max_dataset_size)

    return dataloader.load_data()
