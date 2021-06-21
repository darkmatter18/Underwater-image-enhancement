import collections
from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from utils.SaveObjects import SaveObject


class BaseModel(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.device = torch.device("cuda" if opt.use_cuda else "cpu")
        self.use_cuda = opt.use_cuda
        self.is_distributed = opt.is_distributed

        self.saveMethod = SaveObject(opt)

        # Models
        self.G_AtoB = None
        self.G_BtoA = None
        if self.isTrain:
            self.D_A = None
            self.D_B = None
            self.criterionGAN = None
            self.criterionCycle = None
            self.optimizer_G = None
            self.optimizer_D = None
            self.schedulers = []

        # dataset Variables
        self.real_A = None  # the real images of domain A
        self.real_B = None  # the real images of domain B
        self.fake_A = None  # the fake images of domain A
        self.fake_B = None  # the fake images of domain B
        self.rec_A = None  # the recirculated images of domain A
        self.rec_B = None  # the recirculated images of domain B
        self.image_paths = None  # the path of images of domain A and domain B

        # Loss variables
        self.loss_D_A = None  # Discriminator loss for Discriminator A
        self.loss_D_B = None  # Discriminator loss for Discriminator B
        self.loss_G_AtoB = None  # Generator loss for Generator AtoB
        self.loss_G_BtoA = None  # Generator loss for Generator BtoA
        self.loss_G = None  # Overall Generator loss
        self.cycle_loss_A = None  # Cycle loss for path XtoYtoX
        self.cycle_loss_B = None  # Cycle loss for path XtoYtoX

        # Model Names
        self.net_names = []

    @abstractmethod
    def feed_input(self, x):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    def build_model(self, model: nn.Module):
        model = model.to(self.device)
        if self.is_distributed and self.use_cuda:
            # multi-machine multi-gpu case
            return torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False)
        else:
            # single-machine multi-gpu case or single-machine or multi-machine cpu case
            return torch.nn.DataParallel(model)

    def train(self):
        """
        Set the Models in the train mode
        """
        for model in self.net_names:
            net = getattr(self, model)
            net.train()

    def eval(self) -> None:
        """
        Set the Models in the eval mode
        """
        for model in self.net_names:
            net = getattr(self, model)
            net.eval()

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

    def get_current_image_path(self):
        """
        :return: The current image path
        """
        return self.image_paths

    def get_current_visuals(self) -> collections.OrderedDict:
        """Get the Current Produced Images
        :return: Image artifacts of the last epochs {real_A, real_B, fake_A, fake_B, rec_A, rec_B}
        """

        return collections.OrderedDict({'real_A': self.real_A, 'real_B': self.real_B, 'fake_A': self.fake_A,
                                        'fake_B': self.fake_B, 'rec_A': self.rec_A, 'rec_B': self.rec_B})

    def get_current_losses(self) -> dict:
        """Get the Current Losses

        :return: Losses
        """
        return_dict: dict = {}
        for loss_name in ['loss_D_A', 'loss_D_B', 'loss_G_AtoB', 'loss_G_BtoA',
                          'cycle_loss_A', 'cycle_loss_B']:
            loss = getattr(self, loss_name)
            if isinstance(loss, Tensor):
                _l = loss.item()
            else:
                _l = loss

            return_dict[loss_name] = _l
        return return_dict

    def save_networks(self, epoch: int) -> None:
        """Save all models mentioned in net_names
        :param epoch: Current Epoch (prefix for the name)
        """
        for net_name in self.net_names:
            net = getattr(self, net_name)
            self._save_network(net, net_name, epoch)

    def _save_network(self, net: nn.Module, net_name: str, epoch: int) -> None:
        """
        Responsible to save individual networks
        :param net: The Network
        :param net_name: The network name
        :param epoch: The Epoch
        :return:
        """
        save_filename = f"{epoch}_net_{net_name}"

        if self.use_cuda:
            self.saveMethod.save_model(net, save_filename)
            net.cuda()
        else:
            self.saveMethod.save_model(net, save_filename)

    def save_optimizers_and_scheduler(self, epoch: int):
        """Save optimizers

        :param epoch: Current Epoch (prefix for the name)
        """
        # Saving Optimizers
        self.saveMethod.save_optim_scheduler(self.optimizer_G, f"{epoch}_optim_G.pt")
        self.saveMethod.save_optim_scheduler(self.optimizer_D, f"{epoch}_optim_D.pt")

        # Saving Schedulers
        for i, scheduler in enumerate(self.schedulers):
            self.saveMethod.save_optim_scheduler(scheduler, f"{epoch}_scheduler_{i}.pt")
            
    def update_learning_rate(self, metric=0):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))
