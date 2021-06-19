import os
import torch


class SaveObject:
    def __init__(self, opt):
        self.opt = opt
        self.provider = opt.cloud

    def save_model(self, model, file_name):
        if self.provider == "aws":
            path = os.path.join(self.opt.model_dir, file_name)
            # recommended way from http://pytorch.org/docs/master/notes/serialization.html
            torch.save(model.cpu().state_dict(), path)
        else:
            raise RuntimeError('Save provider can be only aws sagemaker')

    def save_file(self):
        pass

    def __str__(self):
        return f"Using {self.provider} as cloud Provider"
