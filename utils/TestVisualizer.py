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
        self.underwater_images = []
        self.restored_images = []
        self.psrns = []
        self.ssims = []
        self.entropy_underwater_imgs = []
        self.entropy_restored_imgs = []
        self.uiqms_underwater_imgs = []
        self.uiqms_restored_imgs = []
        self.uciqes_underwater_images = []
        self.uciqes_restored_images = []
        self.save_path = None

    def display_inference(self):
        if self.visuals:
            e = len(self.underwater_images)
            fig, ax = plt.subplots(e, 4, figsize=(15, 4 * e))
            for i, (underwater_img, restored_img, psnr, ssim, entropy, uiqm_underwater, uiqm_restored,
                    uciqe_underwater, uciqm_restored) in \
                    enumerate(zip(self.underwater_images, self.restored_images, self.psrns, self.ssims,
                                  self.entropy_underwater_imgs,
                                  self.uiqms_underwater_imgs, self.uiqms_restored_imgs,
                                  self.uciqes_underwater_images, self.uciqes_restored_images)):
                ax[i, 0].imshow(underwater_img)
                ax[i, 0].set_title("A type Real Image")
                ax[i, 1].imshow(restored_img)
                ax[i, 1].set_title("B type Fake Image")
                ax[i, 3].axis("off")
                ax[i, 3].invert_yaxis()
                ax[i, 3].text(0.5, 0.5, f"PSNR: {psnr}\nSSIM: {ssim}\nEntropy: {entropy}\n"
                                        f"UIQM Underwater: {uiqm_underwater}\n"
                                        f"UIQM Restored: {uiqm_restored}\n"
                                        f"UCIQM Underwater: {uciqe_underwater}\n"
                                        f"UCIQM Restored: {uciqm_restored}",
                              verticalalignment="top")

            plt.show()

        if self.save_artifacts:
            self.save_path = os.path.join(os.getcwd(), "output", "images")
        if self.all:
            self.save_path = os.path.join(os.getcwd(), "output", "metrics")

        if self.all or self.save_artifacts:
            mkdirs(self.save_path)
            for i, (underwater_img, restored_img) in enumerate(zip(
                    self.underwater_images, self.restored_images)):
                mpimg.imsave(os.path.join(self.save_path, f"real_A_{self.opt.load_model}_{i}.jpg"), underwater_img)
                mpimg.imsave(os.path.join(self.save_path, f"fake_A_{self.opt.load_model}_{i}.jpg"), restored_img)
            with open(os.path.join(self.save_path, f"{self.opt.load_model}_data.pkl"), 'wb+') as f:
                pickle.dump({'psnr': self.psrns, 'ssim': self.ssims,
                             'entropy_underwater': self.entropy_underwater_imgs,
                             'entropy_restored': self.entropy_restored_imgs,
                             'uiqm_underwater': self.uiqms_underwater_imgs, 'uiqm_restored': self.uiqms_restored_imgs,
                             'uciqm_underwater': self.uciqes_underwater_images,
                             'uciqm_restored': self.uciqes_restored_images,
                             }, f)

    def add_inference(self, image_data: dict, image_path: dict):
        """Displays the test image data

        :param image_data: Image data from Model
        :param image_path: Image path from model
        """
        if 'fake_B' in image_data:
            underwater_image = tensor2im(image_data['real_A'])
            reconstructed_image = tensor2im(image_data['fake_B'])
            try:

                psnr = peak_signal_noise_ratio(underwater_image, reconstructed_image)
                ssim = structural_similarity(underwater_image, reconstructed_image, multichannel=True)

                # Calculate only if --all_metrics is passed explicitly
                entropy_restored = None
                entropy_underwater = None
                uiqm_restored = None
                uiqm_underwater = None
                uciqe_restored = None
                uciqe_underwater = None
                if self.opt.all_metrics:
                    entropy_restored = shannon_entropy(reconstructed_image)
                    uiqm_restored, uciqe_restored = nmetrics(reconstructed_image)
                    entropy_underwater = shannon_entropy(underwater_image)
                    uiqm_underwater, uciqe_underwater = nmetrics(underwater_image)

                self.underwater_images.append(underwater_image)
                self.restored_images.append(reconstructed_image)
                self.psrns.append(psnr)
                self.ssims.append(ssim)
                self.entropy_restored_imgs.append(entropy_restored)
                self.entropy_underwater_imgs.append(entropy_underwater)
                self.uiqms_restored_imgs.append(uiqm_restored)
                self.uiqms_underwater_imgs.append(uiqm_underwater)
                self.uciqes_restored_images.append(uciqe_restored)
                self.uciqes_underwater_images.append(uciqe_underwater)

            except FileNotFoundError as e:
                self.opt.logger.error(f"{image_path['a'][0]} File Not Found, {e}")
            except Exception as exp:
                self.opt.logger.error(exp)
