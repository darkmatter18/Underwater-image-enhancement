import os
import torch


class SaveObject:
    def __init__(self, opt):
        self.provider = opt.cloud

    def save_model(self, model, model_dir):
        path = os.path.join(model_dir, "model.pth")
        # recommended way from http://pytorch.org/docs/master/notes/serialization.html
        torch.save(model.cpu().state_dict(), path)

    def save_file(self):
        pass

    def __str__(self):
        return f"Using {self.provider} as cloud Provider"
