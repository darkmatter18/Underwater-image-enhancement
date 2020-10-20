import os
import torch
import numpy as np
from google.cloud import storage


def mkdirs(paths):
    """create empty directories if they don't exist

    :param paths: a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    :param path: a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def setup_cloud_bucket(dataroot: str):
    """Setup Google Cloud Bucket

    :param dataroot: The Root of the Data-storage
    :return: Bucket
    """
    bucket_name = dataroot.split("/")[2]
    print(f"Using Bucket: {bucket_name} for storing and Loading")
    c = storage.Client()
    b = c.get_bucket(bucket_name)
    assert b.exists(), f"Bucket {bucket_name} dos't exist. Try different one"
    return b


def tensor2im(input_image, imtype=np.uint8):
    """Converts a Tensor array into a numpy image array.

    :param input_image: the input image tensor array
    :param imtype: the desired type of the converted numpy array
    :return: Converted Image
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))

        # post-processing: transpose and scaling
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)
