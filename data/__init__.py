from .dataloaders import UWIEDataloader
from .datasets import UWIEDataset, UWIEDatasetTest


def create_dataset(dataroot: str, subdir: str, phase: str, serial_batches: bool, preprocess: str, no_flip: bool,
                   load_size: int, crop_size: int, batch_size: int, is_distributed: bool, use_cuda: bool,
                   is_test: bool):
    """Create a Dataset using opt

    :param is_test:
    :param subdir:
    :param use_cuda: Whether using CUDA or not
    :param is_distributed: Whether using Distributed training or not
    :param batch_size: Batch Size in Dataloader
    :param crop_size: Crop Size of the images
    :param load_size: Load SIze of the images
    :param no_flip: Whether using Flip or not in preprocessing
    :param preprocess: Type of preprocessing [RAC - Resize and CenterCrop, RRC - Random Resized Crop]
    :param serial_batches: Whether using serial batches or not [Paired images]
    :param phase: The phase of the data
    :param dataroot: The overall Dataroot
    :return: Loaded torch.tensor dataset {'A': A, 'B': B, 'A_paths': a_path, 'B_paths': b_path}
    """
    if is_test:
        dataset = UWIEDatasetTest(dataroot=dataroot, subdir=subdir, phase=phase, preprocess=preprocess,
                                  load_size=load_size, crop_size=crop_size)
    else:
        dataset = UWIEDataset(dataroot=dataroot, subdir=subdir, phase=phase, serial_batches=serial_batches,
                              preprocess=preprocess, flip=not no_flip, load_size=load_size, crop_size=crop_size)

    dataloader = UWIEDataloader(dataset, batch_size=batch_size, is_distributed=is_distributed, use_cuda=use_cuda,
                                is_test=is_test)

    return dataloader.load_data()
