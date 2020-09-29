from .BaseOptions import BaseOptions


class TrainOptions(BaseOptions):
    def initialized(self, parser):
        parser = BaseOptions.initialized(self, parser)
        return parser
