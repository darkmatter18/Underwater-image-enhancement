import argparse
import os

import time


from utils import mkdirs
from utils.SaveObjects import SaveObject

class BaseOptions:
    def __init__(self):
        self.isTrain = False
        self.parser = argparse.ArgumentParser(prog='Underwater Image Enhancement',
                                              description='A Deep Learning based application that enhances '
                                                          'Underwater Images',
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser = self.initialized(self.parser)

    def initialized(self, parser: argparse.ArgumentParser):

        # basic parameters
        # parser.add_argument('--job-dir', dest="checkpoints_dir", type=str, default='./checkpoints',
        #                     help='models are saved here')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--name', type=str, default='uwie',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--name_time', action='store_true', help='Add Timestamp after name')
        parser.add_argument('--model', type=str, required=True, help='Name of the model')
        parser.add_argument('--dataroot', required=True, type=str,
                            help="ROOT of the image dataset (should have sub folders trainA, trainB, valA, valB, etc)")
        # parser.add_argument('--num_gpus', type=int, default=0, help="Number of GPUS for training")
        # parser.add_argument('--hosts', type=list, default=[], help="Hosts list for distributed training")

        # model parameters
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
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--preprocess', type=str, required=True,
                            help='scaling and cropping of images at load time '
                                 '[RRC (RandomResizedCrop) | RAC(Resize and Crop)] Use RRC on training, RAC on testing')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do flip the images for data augmentation')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

        # parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        # parser.add_argument('--backend', type=str, default=None,
        #                     help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu')

        # Cloud parameter
        parser.add_argument('--cloud', default='none', type=str, help="Name of the cloud provider [aws | gcp | none]")
        return parser

    def _print_options(self, opt):
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

        # Add Time stamp in the name
        if self.isTrain and opt.name_time:
            opt.name = opt.name + str(int(time.time()))

        # opt.is_distributed = len(opt.hosts) > 1 and opt.backend is not None
        # opt.use_cuda = opt.num_gpus > 0
        # opt.device = torch.device("cuda" if opt.use_cuda else "cpu")

        opt.save = SaveObject(opt)
        self._print_options(opt)

        return opt
