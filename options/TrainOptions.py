from .BaseOptions import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.isTrain = True

    def initialized(self, parser):
        parser = BaseOptions.initialized(self, parser)
        parser.add_argument('--gan_mode', type=str, default='lsgan',
                            help='the type of GAN objective. [vanilla| lsgan | wgangp]. '
                                 'vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')

        return parser
