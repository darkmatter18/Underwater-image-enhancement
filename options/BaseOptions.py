import os
import argparse
from utils import mkdirs


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog='Underwater Image Enhancement',
                                         description='A very deep application that enhances Underwater Images',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.initialized(self.parser)

    def initialized(self, parser):
        """
        :param parser:
        :return:
        """
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--name', type=str, default='experiment_name',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--dataroot', required=True, type=str,
                            help="path to images (should have sub folders trainA, trainB, valA, valB, etc)")
        parser.add_argument('--phase', default="train", type=str, help="train, val, test, etc")
        parser.add_argument('--max_dataset_size', default=float('inf'), type=float,
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains '
                                 'more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--direction', default="AtoB", type=str, help='AtoB or BtoA')
        parser.add_argument('--input_nc', default=3, type=int,
                            help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3,
                            help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                            help='scaling and cropping of images at load time [resize_and_crop | crop | none]')
        parser.add_argument('--flip', action='store_true',
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
        self.print_options(opt)

        return opt
