import collections
import os
from typing import List

import itertools
import torch
from torch import nn

from utils.image_pool import ImagePool
from .networks import build_D, build_G, get_scheduler, GANLoss


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

    def feed_input(self, x):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        :type x: dict
        :param x: include the data itself and its metadata information.
        x should have the structure {'A': Tensor Images, 'B': Tensor Images,
        'A_paths': paths of the A Images, 'B_paths': paths of the B Images}

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = x['A' if AtoB else 'B'].to(self.device)
        self.real_B = x['B' if AtoB else 'A'].to(self.device)
        self.image_paths = x['A_paths' if AtoB else 'B_paths']

    def optimize_parameters(self):
        # Forward
        self.forward()

        # Train Generators
        self._set_requires_grad([self.D_A, self.D_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

        # Train Discriminators
        self._set_requires_grad([self.D_A, self.D_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

    def forward(self):
        """Run forward pass
        Called by both functions <optimize_parameters> and <test>
        """
        self.fake_B = self.G_AtoB(self.real_A)  # G_A(A)
        self.rec_A = self.G_BtoA(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.G_BtoA(self.real_B)  # G_B(B)
        self.rec_B = self.G_AtoB(self.fake_A)  # G_A(G_B(B))

    def backward_G(self):
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # GAN loss D_A(G_AtoB(A))
        self.loss_G_AtoB = self.criterionGAN(self.D_A(self.fake_B), True)

        # GAN loss D_B(G_BtoA(B))
        self.loss_G_BtoA = self.criterionGAN(self.D_B(self.fake_A), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.cycle_loss_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

        # Backward cycle loss || G_A(G_B(B)) - B||
        self.cycle_loss_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_AtoB + self.loss_G_BtoA + self.cycle_loss_A + self.cycle_loss_B
        self.loss_G += self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        :param netD: the discriminator D
        :param real: real images
        :param fake: images generated by a generator
        :return: Loss
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.D_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.D_B, self.real_A, fake_A)

    def _set_requires_grad(self, nets: List[nn.Module], requires_grad: bool = False) -> None:
        """
        Set requires_grad=False for all the networks to avoid unnecessary computations
        :param nets: a list of networks
        :param requires_grad: whether the networks require gradients or not
        """
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train(self):
        """Make models train mode during test time"""
        self.G_AtoB.train()
        self.G_BtoA.train()

        if self.isTrain:
            self.D_A.train()
            self.D_B.train()

    def eval(self):
        """Make models eval mode during test time"""
        self.G_AtoB.eval()
        self.G_BtoA.eval()

        if self.isTrain:
            self.D_A.eval()
            self.D_B.eval()

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

    def save_networks(self, epoch):
        """Save models

        :type epoch: str
        :param epoch: Current Epoch (prefix for the name)
        """
        for net_name in self.net_names:
            net = getattr(self, net_name)
            self.save_network(net, net_name, epoch)

    def save_optimizers_and_scheduler(self, epoch):
        """Save optimizers

        :type epoch: str
        :param epoch: Current Epoch (prefix for the name)
        """
        # Saving Optimizers
        self.save_optimizer_scheduler(self.optimizer_G, f"{epoch}_optim_G.pt")
        self.save_optimizer_scheduler(self.optimizer_D, f"{epoch}_optim_D.pt")

        # Saving Schedulers
        self.save_optimizer_scheduler(self.schedulers[0], f"{epoch}_scheduler_0.pt")
        self.save_optimizer_scheduler(self.schedulers[1], f"{epoch}_scheduler_1.pt")

    def save_optimizer_scheduler(self, optim_or_scheduler, name):
        """Save a single optimizer

        :param optim_or_scheduler: The optimizer object
        :type name: str
        :param name: Name of the optimizer
        """
        save_path = os.path.join(self.save_dir, name)

        torch.save(optim_or_scheduler.state_dict(), save_path)

    def save_network(self, net, net_name, epoch):
        save_filename = '%s_net_%s.pt' % (epoch, net_name)
        if self.opt.isCloud:
            save_path = save_filename
        else:
            save_path = os.path.join(self.save_dir, save_filename)

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(net.module.cpu().state_dict(), save_path)
            net.cuda(self.gpu_ids[0])
        else:
            torch.save(net.cpu().state_dict(), save_path)

    def get_current_losses(self) -> dict:
        """Get the Current Losses

        :return: Losses
        """
        if isinstance(self.loss_idt_A, (int, float)):
            idt_loss_A = self.loss_idt_A
        else:
            idt_loss_A = self.loss_idt_A.item()

        if isinstance(self.loss_idt_B, (int, float)):
            idt_loss_B = self.loss_idt_B
        else:
            idt_loss_B = self.loss_idt_B.item()
        return collections.OrderedDict({'loss_idt_A': idt_loss_A, 'loss_idt_B': idt_loss_B,
                                        'loss_D_A': self.loss_D_A.item(), 'loss_D_B': self.loss_D_B.item(),
                                        'loss_G_AtoB': self.loss_G_AtoB.item(), 'loss_G_BtoA': self.loss_G_BtoA.item(),
                                        'cycle_loss_A': self.cycle_loss_A.item(),
                                        'cycle_loss_B': self.cycle_loss_B.item()})

    def get_current_image_path(self):
        """
        :return: The current image path
        """
        return self.image_paths

    def get_current_visuals(self):
        """Get the Current Produced Images

        :return: Images {real_A, real_B, fake_A, fake_B, rec_A, rec_B}
        :rtype: dict
        """
        r = collections.OrderedDict({'real_A': self.real_A, 'real_B': self.real_B})

        if self.fake_A is not None:
            r['fake_A'] = self.fake_A
        if self.fake_B is not None:
            r['fake_B'] = self.fake_B
        if self.rec_A is not None:
            r['rec_A'] = self.rec_A
        if self.rec_B is not None:
            r['rec_B'] = self.rec_B
        return r