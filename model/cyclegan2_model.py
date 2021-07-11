import torch
import itertools

from .BaseModel import BaseModel
from networks.cyclegan2_networks import NLayerDiscriminator, ResnetGenerator, GANLoss, get_scheduler, get_norm_layer


class CycleGan2Model(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.G_AtoB = self.build_model(ResnetGenerator(input_nc=3, output_nc=3, ngf=self.opt.ngf,
                                                       n_blocks=self.opt.n_blocks_G,
                                                       norm_layer=get_norm_layer(self.opt.norm),
                                                       use_dropout=not self.opt.no_dropout))
        self.G_BtoA = self.build_model(ResnetGenerator(input_nc=3, output_nc=3, ngf=self.opt.ngf,
                                                       n_blocks=self.opt.n_blocks_G,
                                                       norm_layer=get_norm_layer(self.opt.norm),
                                                       use_dropout=not self.opt.no_dropout))
        self.net_names = ['G_AtoB', 'G_BtoA']
        if self.isTrain:
            self.D_A = self.build_model(NLayerDiscriminator(input_nc=3, ndf=self.opt.ndf,
                                                            norm_layer=get_norm_layer(self.opt.norm),
                                                            n_layers=self.opt.n_layers_D))
            self.D_B = self.build_model(NLayerDiscriminator(input_nc=3, ndf=self.opt.ndf,
                                                            norm_layer=get_norm_layer(self.opt.norm),
                                                            n_layers=self.opt.n_layers_D))

            self.net_names.append('D_A')
            self.net_names.append('D_B')

            self.criterionGAN = GANLoss(self.opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss().to(self.device)

            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.G_AtoB.parameters(), self.G_BtoA.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.schedulers = [get_scheduler(optimizer, lr_policy=self.opt.lr_policy, n_epochs=self.opt.n_epochs,
                                             lr_decay_iters=self.opt.lr_decay_iters, epoch_count=self.opt.epoch_count,
                                             n_epochs_decay=self.opt.n_epochs_decay) for optimizer in self.optimizers]

        if self.isTrain:
            if self.opt.ct > 0:
                self.opt.logger.info(f"Continue training from {self.opt.ct} Localing Model from {self.opt.ct-1}")
                self.load_train_model(str(self.opt.ct-1))

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
        self.image_paths = {"a": x['A_paths'], "b": x["B_paths"]}

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
        self.loss_G.backward()

    def backward_D_basic(self, netD: torch.nn.Module, real: torch.Tensor, fake: torch.Tensor):
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
        self.loss_D_A = self.backward_D_basic(self.D_A, self.real_B, self.fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        self.loss_D_B = self.backward_D_basic(self.D_B, self.real_A, self.fake_A)
