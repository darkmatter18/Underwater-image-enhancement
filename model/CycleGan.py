import torch
import itertools
from utils.image_pool import ImagePool
from .networks import build_D, build_G, get_scheduler, GANLoss


class CycleGan:
    def __init__(self, opt):
        # Initialize the Models

        # Global Variables
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
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

        if self.isTrain:
            self.D_A = build_D(input_nc=opt.output_nc, ndf=opt.ndf, n_layers=opt.n_layers_D, norm=opt.norm,
                               init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)
            self.D_B = build_D(input_nc=opt.input_nc, ndf=opt.ndf, n_layers=opt.n_layers_D, norm=opt.norm,
                               init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)

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



    def forward(self):
        """Run forward pass
        Called by both functions <optimize_parameters> and <test>
        """
        self.fake_B = self.G_AtoB(self.real_A)  # G_A(A)
        self.rec_A = self.G_BtoA(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.G_BtoA(self.real_B)  # G_B(B)
        self.rec_B = self.G_AtoB(self.fake_A)  # G_A(G_B(B))

    def compute_identity_loss(self):
        """Compute the Identity Loss

        :return: Identity Loss
        """
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
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

        return self.loss_idt_A + self.loss_idt_B

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
