import os
from typing import List

import itertools
import torch

from utils.image_pool import ImagePool
from networks.cyclegan1_networks import build_D, build_G, get_scheduler, GANLoss


class CycleGan:
    def __init__(self, opt):
        # Initialize the Models

        # Global Variables
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        self.device = torch.device(f'cuda:{self.gpu_ids[0]}') if self.gpu_ids else torch.device('cpu')
        self.metric = 0  # used for learning rate policy 'plateau'

        self.G_AtoB = build_G(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=opt.ngf, norm=opt.norm,
                              padding_type=opt.padding_type,
                              use_dropout=not opt.no_dropout, n_blocks=opt.n_blocks_G, init_type=opt.init_type,
                              init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)

        self.G_BtoA = build_G(input_nc=opt.output_nc, output_nc=opt.input_nc, ngf=opt.ngf, norm=opt.norm,
                              padding_type=opt.padding_type,
                              use_dropout=not opt.no_dropout, n_blocks=opt.n_blocks_G, init_type=opt.init_type,
                              init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)

        self.net_names = ['G_AtoB', 'G_BtoA']

        if self.isTrain:
            self.D_A = build_D(input_nc=opt.output_nc, ndf=opt.ndf, n_layers=opt.n_layers_D, norm=opt.norm,
                               init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)
            self.D_B = build_D(input_nc=opt.input_nc, ndf=opt.ndf, n_layers=opt.n_layers_D, norm=opt.norm,
                               init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)

            self.net_names.append('D_A')
            self.net_names.append('D_B')

            # create image buffer to store previously generated images
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_AtoB.parameters(), self.G_BtoA.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # lr Scheduler
            self.schedulers = [get_scheduler(optimizer, lr_policy=opt.lr_policy, n_epochs=opt.n_epochs,
                                             lr_decay_iters=opt.lr_decay_iters, epoch_count=opt.epoch_count,
                                             n_epochs_decay=opt.n_epochs_decay) for optimizer in self.optimizers]

        # Internal Variables
        self.real_A = None
        self.real_B = None
        self.image_paths = None
        self.fake_A = None
        self.fake_B = None
        self.rec_A = None
        self.rec_B = None
        self.idt_A = None
        self.idt_B = None
        self.loss_idt_A = None
        self.loss_idt_B = None
        self.loss_G_AtoB = None
        self.loss_G_BtoA = None
        self.cycle_loss_A = None
        self.cycle_loss_B = None
        self.loss_G = None
        self.loss_D_A = None
        self.loss_D_B = None

        # Printing the Networks
        for net_name in self.net_names:
            print(net_name, "\n", getattr(self, net_name))

        # Continue training, if isTrain
        if self.isTrain:
            if self.opt.ct > 0:
                print(f"Continue training from {self.opt.ct}")
                self.load_train_model(str(self.opt.ct))

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))


    def compute_visuals(self, bidirectional=False):
        """ Computes the Visual output data from the model
        :type bidirectional: bool
        :param bidirectional: if true, Calculate both AtoB and BtoA, else calculate AtoB
        """
        self.eval()
        with torch.no_grad():
            self.fake_B = self.G_AtoB(self.real_A)
            if bidirectional:
                self.fake_A = self.G_BtoA(self.real_B)

    def _load_objects(self, file_names: List[str], object_names: List[str]):
        """Load objects from file

        :param file_names: Name of the Files to load
        :param object_names: Name of the object, where the files is going to be stored.

        file_names and object_names should be in same order
        """
        for file_name, object_name in zip(file_names, object_names):
            model_name = os.path.join(self.save_dir, file_name)
            print(f"Loading {object_name} from {model_name}")
            state_dict = torch.load(model_name, map_location=self.device)

            net = getattr(self, object_name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            net.load_state_dict(state_dict)

    def load_networks(self, initials, load_D=False):
        """ Loading Models
        Loads from /checkpoint_dir/name/{initials}_net_G_AtoB.pt
        :type initials: str
        :param initials: The initials of the model
        :type load_D: bool
        :param load_D: Is loading D or not
        """
        file_names = [f"{initials}_net_G_AtoB.pt", f"{initials}_net_G_BtoA.pt"]
        if load_D:
            file_names.append(f"{initials}_net_D_A.pt")
            file_names.append(f"{initials}_net_D_B.pt")

        object_names = ['G_AtoB', 'G_BtoA'] if not load_D else ['G_AtoB', 'G_BtoA', 'D_A', 'D_B']

        self._load_objects(file_names, object_names)

    def load_lr_schedulers(self, initials):
        s_file_name_0 = os.path.join(self.save_dir, f"{initials}_scheduler_0.pt")
        s_file_name_1 = os.path.join(self.save_dir, f"{initials}_scheduler_1.pt")

        print(f"Loading scheduler-0 from {s_file_name_0}")
        self.schedulers[0].load_state_dict(torch.load(s_file_name_0, map_location=self.device))
        print(f"Loading scheduler-1 from {s_file_name_1}")
        self.schedulers[1].load_state_dict(torch.load(s_file_name_1, map_location=self.device))

    def load_train_model(self, initials):
        """ Loading Models for training purpose

        :type initials: str
        :param initials: Initials of the object names
        """
        self.load_networks(initials, load_D=True)

        optim_file_names = [f"{initials}_optim_G.pt", f"{initials}_optim_D.pt"]
        optim_object_names = ['optimizer_G', 'optimizer_D']

        self._load_objects(optim_file_names, optim_object_names)

        self.load_lr_schedulers(initials)



