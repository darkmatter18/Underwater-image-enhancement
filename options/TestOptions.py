from .BaseOptions import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.isTrain = False

    def initialized(self, parser):
        parser = BaseOptions.initialized(self, parser)

        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--load_model', type=str, default='latest', help="name of the models to load")
        return parser
