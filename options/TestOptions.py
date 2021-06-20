import os
from .BaseOptions import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.isTrain = False

    def initialized(self, parser):
        parser = BaseOptions.initialized(self, parser)

        parser.add_argument('--phase', type=str, default='test')
        parser.add_argument('--training-data-dir', type=str, default=os.getenv("SM_CHANNEL_TRAINING"),
                            help="Training data directory")
        parser.add_argument('--load_model', type=str, default='latest', help="name of the models to load")
        parser.add_argument('--examples', type=int, default=1, help='No of examples for testing')
        return parser
