import torch


class SaveObject:
    def __init__(self, opt):
        self.provider = opt.cloud
        if self.provider == "aws":
            # Do AWS related stuff
            pass
        elif self.provider == "gcp":
            # Do GCP related stuff
            pass
        else:
            # Do ofr local
            pass

    def save_torch_artifacts(self, artifact: dict, save_path: str) -> None:
        torch.save(artifact, save_path)
        if self.provider == "gcp":
            # Do after saving works here
            pass

    def save_file(self):
        pass

    def __str__(self):
        return f"Using {self.provider} as cloud Provider"
