import os
import time
import torch
import argparse
from utils import mkdirs
from torch import distributed


class BaseOptions:
    def __init__(self):
        self.isTrain = False
        self.parser = argparse.ArgumentParser(prog='Underwater Image Enhancement',
                                              description='A very deep application that enhances Underwater Images',
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.initialized(self.parser)

    def initialized(self, parser):
        """
        :param parser:
        :return:
        """
        # basic parameters
        parser.add_argument('--job-dir', dest="checkpoints_dir", type=str, default='./checkpoints',
                            help='models are saved here')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--name', type=str, default='uwie',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--dataroot', required=True, type=str,
                            help="path to images (should have sub folders trainA, trainB, valA, valB, etc)")
        parser.add_argument('--no_gpu', action='store_true', help='Use only CPU')

        # model parameters
        parser.add_argument('--input_nc', default=3, type=int,
                            help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3,
                            help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--n_blocks_G', type=int, default=9, help='no of Resnet Blocks in Generator')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--padding_type', type=str, default='reflect',
                            help='the name of padding layer: reflect | replicate | zero')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')

        # dataset parameters
        parser.add_argument('--max_dataset_size', default=float('inf'), type=float,
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains '
                                 'more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--direction', default="AtoB", type=str, help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                            help='scaling and cropping of images at load time [resize_and_crop | crop | none]')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do flip the images for data augmentation')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')

        return parser

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt = self.parser.parse_args()
        opt.isTrain = self.isTrain
        if self.isTrain:
            opt.name = opt.name + str(int(time.time()))

        # IsCloud setup
        if opt.dataroot.startswith('gs://'):
            opt.isCloud = True
            opt.bucket_name = opt.dataroot.split("/")[2]
            opt.dataroot = "/".join(opt.dataroot.split("/")[3:])
            opt.checkpoints_dir = "/".join(opt.checkpoints_dir.split("/")[3:])
        else:
            opt.isCloud = False

        self.print_options(opt)

        # set gpu ids
        if not opt.no_gpu and torch.cuda.is_available():
            device_ids = list(range(torch.cuda.device_count()))
            gpus = len(device_ids)
            print(f'{gpus} no of GPUs detected. Using GPU: {str(device_ids)}')
            torch.cuda.set_device(device_ids[0])
            if distributed.is_available():
                # print(f"MASTER_ADDR is: {os.environ['MASTER_ADDR']}")
                # print(f"MASTER_PORT is: {os.environ['MASTER_PORT']}")
                # print(f"RANK is:{os.environ['RANK']}")
                # print(f"World Size is: {os.environ['WORLD_SIZE']}")
                # torch.distributed.init_process_group('nccl', init_method="env://")
                print("Running on Distributed mode")
            else:
                print("Distributed mode is not supported")
        else:
            device_ids = []
            print('No GPU. switching to CPU')
        opt.gpu_ids = device_ids

        return opt
