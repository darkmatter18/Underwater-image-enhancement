import os
from .BaseOptions import BaseOptions


class TestOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.isTrain = False

    def initialized(self, parser):
        parser = BaseOptions.initialized(self, parser)

        parser.add_argument('--test-dataset-dir', type=str, default=os.path.join(os.path.dirname(os.getcwd()),
                                                                                 "Dataset", "EUVP Dataset", "paired"))
        parser.add_argument('--test-subdir', type=str, default="underwater_dark")
        parser.add_argument('--phase', type=str, default='test')
        parser.add_argument('--training-data-dir', type=str, default=os.getenv("SM_CHANNEL_TRAINING"),
                            help="Training data directory")
        parser.add_argument('--load_model', type=str, default='latest', help="name of the models to load")
        parser.add_argument('--all', action='store_true')
        parser.add_argument('--examples', type=int, default=1, help='No of examples for testing')
        parser.add_argument('--visuals', action='store_true')
        parser.add_argument('--all_metrics', action='store_true')
        parser.add_argument('--save_artifacts', action='store_true')
        return parser
