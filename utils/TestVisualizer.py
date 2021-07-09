import os
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from . import tensor2im, mkdirs
from .nmetrics import nmetrics


class TestVisualizer:
    def __init__(self, opt):
        self.opt = opt
        self.dataroot = opt.test_dataset_dir
        self.subdir = opt.test_subdir
        self.phase = opt.phase
        self.visuals = opt.visuals
        self.save_artifacts = opt.save_artifacts
        self.all = opt.all

        self.dir_A = os.path.join(self.dataroot, self.subdir, self.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(self.dataroot, self.subdir, self.phase + 'B')  # create a path '/path/to/data/trainB'

        # Storing Images and paths
        self.real_images = []
        self.fake_images = []
        self.original_of_fake_images = []
        self.path_names = []
        self.psrns = []
        self.ssims = []
        self.entropy = []
        self.uiqms = []
        self.uciqes = []

    def display_inference(self):
        if self.visuals:
            e = len(self.real_images)
            fig, ax = plt.subplots(e, 4, figsize=(15, 4 * e))
            for i, (r_i, f_i, o_f_i, psnr, ssim, path_name, entropy, uiqm, uciqe) in \
                    enumerate(zip(self.real_images, self.fake_images, self.original_of_fake_images, self.psrns,
                                  self.ssims, self.path_names, self.entropy, self.uiqms, self.uciqes)):
                ax[i, 0].imshow(r_i)
                ax[i, 0].set_title("A type Real Image")
                ax[i, 1].imshow(f_i)
                ax[i, 1].set_title("B type Fake Image")
                ax[i, 2].imshow(o_f_i)
                ax[i, 2].set_title("B type Real Image")
                ax[i, 3].axis("off")
                ax[i, 3].invert_yaxis()
                ax[i, 3].text(0.5, 0.5, f"PSNR: {psnr}\nSSIM: {ssim}\nEntropy: {entropy}\nUIQM: {uiqm}\n"
                                        f"UCIQM: {uciqe}\n Path: {path_name}", verticalalignment="top")

            plt.show()

        if self.save_artifacts:
            p = os.path.join(os.getcwd(), "output", "images")
            mkdirs(p)
            for i, (r_i, f_i, o_f_i) in enumerate(zip(
                    self.real_images, self.fake_images, self.original_of_fake_images)):
                mpimg.imsave(os.path.join(p, f"real_A_{self.opt.load_model}_{i}.jpg"), r_i)
                mpimg.imsave(os.path.join(p, f"fake_A_{self.opt.load_model}_{i}.jpg"), f_i)
                mpimg.imsave(os.path.join(p, f"real_B_{self.opt.load_model}_{i}.jpg"), o_f_i)
            with open(os.path.join(p, f"{self.opt.load_model}_data.pkl"), 'wb+') as f:
                pickle.dump({'psnr': self.psrns, 'ssim': self.ssims, 'entropy': self.entropy, 'uiqm': self.uiqms,
                             'uciqm': self.uciqes}, f)

        if self.all:
            p = os.path.join(os.getcwd(), "output", "metrics")
            mkdirs(p)
            with open(os.path.join(p, f"{self.opt.load_model}_data.pkl"), 'wb+') as f:
                pickle.dump({'psnr': self.psrns, 'ssim': self.ssims, 'entropy': self.entropy, 'uiqm': self.uiqms,
                             'uciqm': self.uciqes}, f)

    def add_inference(self, image_data: dict, image_path: dict):
        """Displays the test image data

        :param image_data: Image data from Model
        :param image_path: Image path from model
        """
        if 'fake_B' in image_data:
            real_i = tensor2im(image_data['real_A'])
            fake_i = tensor2im(image_data['fake_B'])
            try:
                original_of_fake_i = mpimg.imread(os.path.join(self.dir_B, os.path.basename(
                    os.path.normpath(image_path["a"][0]))))

                psnr = peak_signal_noise_ratio(original_of_fake_i, fake_i)
                ssim = structural_similarity(original_of_fake_i, fake_i, multichannel=True)

                # Calculate only if --all_metrics is passed explicitly
                entropy = None
                uiqm = None
                uciqe = None
                if self.opt.all_metrics:
                    entropy = shannon_entropy(fake_i)
                    uiqm, uciqe = nmetrics(fake_i)

                self.real_images.append(real_i)
                self.fake_images.append(fake_i)
                self.original_of_fake_images.append(original_of_fake_i)
                self.path_names.append(os.path.basename(os.path.normpath(image_path["a"][0])))
                self.psrns.append(psnr)
                self.ssims.append(ssim)
                self.entropy.append(entropy)
                self.uiqms.append(uiqm)
                self.uciqes.append(uciqe)
            except:
                self.opt.logger.error(f"{image_path['a'][0]} File Not Found")
