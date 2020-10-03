import os
import random
import numpy as np
from PIL import Image
from io import BytesIO
from google.cloud import storage
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        '.tif', '.TIF', '.tiff', '.TIFF',
    ]

    def __init__(self, dataroot, phase, max_dataset_size=float("inf"), direction="AtoB", input_nc=3, output_nc=3,
                 serial_batches=True, preprocess='resize_and_crop', flip=True, load_size=286, crop_size=256):
        """
        Custom Dataset for feeding Image to the network

        :type dataroot: str
        :param dataroot: Root of the Dataset
        :type phase: str
        :param phase: Folder phase for Dataset
        :type max_dataset_size: float
        :param max_dataset_size: Max size of the Dataset. Default: inf
        :type direction: str
        :param direction: direction of the dataflow ["AtoB" | "BtoA" ]. Default: "AtoB"
        :type input_nc int
        :param input_nc: number of channels of input Image. Default: 3
        :type output_nc int
        :param output_nc: number of channels of output Image. Default: 3
        :type serial_batches int
        :param serial_batches: Serial Batches for input. Default: 3
        :type preprocess: str
        :param preprocess: Type of preprocessing applied. Default: "resize_and_crop"
        :type flip: bool
        :param flip: Is RandomHorizontalFlip is applied or not. Default: True
        :type load_size: int
        :param load_size: Size of the image on load. Default: 286
        :type crop_size: int
        :param crop_size: Size of the image after resize. Default: 256
        """

        self.dataroot = dataroot
        self.phase = phase
        self.max_dataset_size = max_dataset_size
        self.direction = direction
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.serial_batches = serial_batches
        self.preprocess = preprocess
        self.flip = flip
        self.load_size = load_size
        self.crop_size = crop_size

        if self.dataroot.startswith('gs://'):
            self.isCloud = True
            # create a path '/path/to/data/trainA'
            self.dir_A = os.path.join("/".join(self.dataroot.split("/")[3:]), self.phase + 'A')
            # create a path '/path/to/data/trainB'
            self.dir_B = os.path.join("/".join(self.dataroot.split("/")[3:]), self.phase + 'B')
            self.bucket = self.setup_cloud_bucket(dataroot)
            self.A_paths = sorted(self.make_cloud_dataset(self.dir_A, self.max_dataset_size))
            self.B_paths = sorted(self.make_cloud_dataset(self.dir_A, self.max_dataset_size))
        else:
            self.isCloud = False
            self.dir_A = os.path.join(self.dataroot, self.phase + 'A')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(self.dataroot, self.phase + 'B')  # create a path '/path/to/data/trainB'
            self.A_paths = sorted(
                self.make_dataset(self.dir_A, self.max_dataset_size))  # load images from '/path/to/data/trainA'
            self.B_paths = sorted(
                self.make_dataset(self.dir_B, self.max_dataset_size))  # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.direction == 'BtoA'
        input_nc = self.output_nc if btoA else self.input_nc  # get the number of channels of input image
        output_nc = self.input_nc if btoA else self.output_nc  # get the number of channels of output image
        self.transform_A = self.get_transform(grayscale=(input_nc == 1))
        self.transform_B = self.get_transform(grayscale=(output_nc == 1))

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

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)

    def make_dataset(self, dataset_dir, max_dataset_size=float("inf")):
        images = []
        assert os.path.isdir(dataset_dir), '%s is not a valid directory' % dataset_dir

        for root, _, fnames in sorted(os.walk(dataset_dir)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images[:min(max_dataset_size, len(images))]

    def setup_cloud_bucket(self, dataroot):
        """Setup Google Cloud Bucket

        :type dataroot: str
        :param dataroot: The Root of the Data-storage
        :return: Bucket
        """
        bucket_name = dataroot.split("/")[2]
        print(f"Using Bucket: {bucket_name} for fetching dataset")
        c = storage.Client()
        b = c.get_bucket(bucket_name)

        assert b.exists(), f"Bucket {bucket_name} dos't exist. Try different one"

        return b

    def make_cloud_dataset(self, dataset_dir, max_dataset_size=float("inf")):
        images = []
        for b in self.bucket.list_blobs(prefix=dataset_dir):
            if self.is_image_file(b.name):
                images.append(b.name)
        return images[:min(max_dataset_size, len(images))]

    def get_transform(self, grayscale=False, convert=True):
        transform_list = []
        if grayscale:
            transform_list.append(transforms.Grayscale(1))

        if 'resize' in self.preprocess:
            transform_list.append(transforms.Resize([self.load_size, self.load_size]))

        if 'crop' in self.preprocess:
            transform_list.append(transforms.RandomCrop(self.crop_size))

        if not self.flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        if convert:
            transform_list += [transforms.ToTensor()]
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
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

        if self.isCloud:
            a_img = Image.open(BytesIO(self.bucket.get_blob(a_path).download_as_string())).convert('RGB')
            b_img = Image.open(BytesIO(self.bucket.get_blob(b_path).download_as_string())).convert('RGB')
        # Open images
        else:
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
