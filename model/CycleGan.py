import os
import torch
import itertools
import collections
from utils.image_pool import ImagePool
from google.cloud import storage
from .networks import build_D, build_G, get_scheduler, GANLoss


class CycleGan:
    def __init__(self, opt):
        # Initialize the Models

        # Global Variables
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.isCloud = opt.checkpoints_dir.startswith('gs://')
        if self.isCloud:
            self.save_dir = os.path.join("/".join(opt.checkpoints_dir.split("/")[3:]), opt.name)
            self.bucket = self.setup_cloud_bucket(opt.checkpoints_dir)
        else:
            self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
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
            # only works when input and output images have the same number of channels
            if opt.lambda_identity > 0.0:
                assert (opt.input_nc == opt.output_nc)

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

        :param x: include the data itself and its metadata information.

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
        self.set_requires_grad([self.D_A, self.D_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights

        # Train Discriminators
        self.set_requires_grad([self.D_A, self.D_B], True)
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
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity Loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.G_AtoB(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.G_AtoB(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

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

    def set_requires_grad(self, nets, requires_grad=False):
        """
        Set requires_grad=False for all the networks to avoid unnecessary computations
        :param nets: a list of networks
        :param requires_grad: whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def eval(self):
        """Make models eval mode during test time"""
        self.G_AtoB.eval()
        self.G_BtoA.eval()

        if self.isTrain:
            self.D_A.eval()
            self.D_B.eval()

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
        :return:
        """
        if self.isCloud:
            save_path = name
        else:
            save_path = os.path.join(self.save_dir, name)

        torch.save(optim_or_scheduler.state_dict(), save_path)

        if self.isCloud:
            self.save_file_to_cloud(os.path.join(self.save_dir, save_path), save_path)
            os.remove(save_path)

    def save_network(self, net, net_name, epoch):
        save_filename = '%s_net_%s.pt' % (epoch, net_name)
        if self.isCloud:
            save_path = save_filename
        else:
            save_path = os.path.join(self.save_dir, save_filename)

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(net.module.cpu().state_dict(), save_path)
            net.cuda(self.gpu_ids[0])
        else:
            torch.save(net.cpu().state_dict(), save_path)

        if self.isCloud:
            self.save_file_to_cloud(os.path.join(self.save_dir, save_path), save_path)
            os.remove(save_path)

    def setup_cloud_bucket(self, dataroot):
        """Setup Google Cloud Bucket

        :type dataroot: str
        :param dataroot: The Root of the Data-storage
        :return: Bucket
        """
        bucket_name = dataroot.split("/")[2]
        print(f"Using Bucket: {bucket_name} for storing artifacts")
        c = storage.Client()
        b = c.get_bucket(bucket_name)

        assert b.exists(), f"Bucket {bucket_name} dos't exist. Try different one"

        return b

    def save_file_to_cloud(self, file_path_cloud, file_path_local):
        self.bucket.blob(file_path_cloud).upload_from_filename(file_path_local)

    def get_current_losses(self):
        """Get the Current Losses
        :return: Losses
        :rtype: dict
        """
        return collections.OrderedDict({'loss_idt_A': self.loss_idt_A.item(), 'loss_idt_B': self.loss_idt_B.item(),
                                        'loss_D_A': self.loss_D_A.item(), 'loss_D_B': self.loss_D_B.item(),
                                        'loss_G_AtoB': self.loss_G_AtoB.item(), 'loss_G_BtoA': self.loss_G_BtoA.item(),
                                        'cycle_loss_A': self.cycle_loss_A.item(),
                                        'cycle_loss_B': self.cycle_loss_B.item()})

    def get_current_visuals(self):
        """Get the Current Produced Images
        :return: Images
        :rtype: dict
        """
        return collections.OrderedDict({'real_A': self.real_A, 'real_B': self.real_B, 'fake_A': self.fake_A,
                                        'fake_B': self.fake_B, 'rec_A': self.rec_A, 'rec_B': self.rec_B})
