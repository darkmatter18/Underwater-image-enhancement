from .BaseOptions import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.isTrain = True

    def initialized(self, parser):
        parser = BaseOptions.initialized(self, parser)
        # Training Stats params
        parser.add_argument('--visuals_freq', type=int, default=1,
                            help='frequency of showing and saving training results (images and logs) on console')
        parser.add_argument('--artifact_freq', type=int, default=1,
                            help='frequency of saving model artifacts at the end of epochs')

        # network saving and loading parameters
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # parser.add_argument('--ct', type=int, default=0, help='Adding continue training. '
        #                                                       'The value is the epoch no, which the model will start '
        #                                                       'training from, and loads the model from.')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, '
                                 'we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>,')

        # Training Parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100,
                            help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan',
                            help='the type of GAN objective. [vanilla| lsgan | wgangp]. '
                                 'vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50,
                            help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        parser.add_argument('--lambda_identity', type=float, default=0.0,
                            help='use identity mapping. Setting lambda_identity other than 0 has an effect of '
                                 'scaling the weight of the identity mapping loss. '
                                 'For example, if the weight of the identity loss should be 10 times smaller '
                                 'than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser
