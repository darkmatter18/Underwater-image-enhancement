import os
import random

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class UWIEDataset(Dataset):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        '.tif', '.TIF', '.tiff', '.TIFF',
    ]

    def __init__(self, dataroot: str, serial_batches: bool = True, preprocess: str = 'RRC',
                 flip: bool = True, load_size: int = 256, crop_size: int = 224):
        """
        Custom Dataset for feeding Image to the network

        :param dataroot: Root of the Dataset
        :param serial_batches: Serial Batches for input. Default: 3
        :param preprocess: Type of preprocessing applied. Default: "resize_and_crop"
        :param flip: Is RandomHorizontalFlip is applied or not. Default: True
        :param load_size: Size of the image on load. Default: 286
        :param crop_size: Size of the image after resize. Default: 256
        """

        self.dataroot = dataroot
        self.serial_batches = serial_batches
        self.preprocess = preprocess
        self.flip = flip
        self.load_size = load_size
        self.crop_size = crop_size

        self.dir_A = self.dataroot + 'A'  # create a path '/path/to/data/trainA'
        self.dir_B = self.dataroot + 'B'  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(self.make_dataset(self.dir_A))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(self.make_dataset(self.dir_B))  # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.transform_A = self.get_transform()
        self.transform_B = self.get_transform()

    def get_params(self, size):
        w, h = size
        new_h = h
        new_w = w
        if self.preprocess == 'resize_and_crop':
            new_h = new_w = self.load_size

        x = random.randint(0, np.maximum(0, new_w - self.crop_size))
        y = random.randint(0, np.maximum(0, new_h - self.crop_size))

        flip = random.random() > 0.5

        return {'crop_pos': (x, y), 'flip': flip}

    def _is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)

    def make_dataset(self, dataset_dir):
        images = []
        assert os.path.isdir(dataset_dir), '%s is not a valid directory' % dataset_dir

        for root, _, fnames in sorted(os.walk(dataset_dir)):
            for fname in fnames:
                if self._is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images

    def get_transform(self):
        transform_list = []

        # Resize And Crop
        if 'RAC' == self.preprocess:
            transform_list.append(transforms.Resize([self.load_size, self.load_size]))
            transform_list.append(transforms.CenterCrop(self.crop_size))

        elif 'RRC' == self.preprocess:
            transform_list.append(transforms.RandomResizedCrop(self.crop_size))

        if self.flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        :param index: a random integer for data indexing
        :return: Returns a dictionary that contains A, B, A_paths and B_paths
                    A (tensor)       -- an image in the input domain
                    B (tensor)       -- its corresponding image in the target domain
                    A_paths (str)    -- image paths
                    B_paths (str)    -- image paths
        """

        a_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.serial_batches:  # make sure index is within then range
            index_b = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_b = random.randint(0, self.B_size - 1)
        b_path = self.B_paths[index_b]

        a_img = Image.open(a_path).convert('RGB')
        b_img = Image.open(b_path).convert('RGB')

        # apply image transformation
        A = self.transform_A(a_img)
        B = self.transform_B(b_img)

        return {'A': A, 'B': B, 'A_paths': a_path, 'B_paths': b_path}

    def __len__(self):
        """
        As we have two datasets with potentially different number of images,
        we take a maximum of them.

        :return: the total number of images in the dataset.
        """
        return max(self.A_size, self.B_size)
