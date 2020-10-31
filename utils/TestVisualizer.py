import os
import matplotlib.pyplot as plt
from . import tensor2im
import matplotlib.image as mpimg
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class TestVisualizer:
    def __init__(self, opt):
        self.opt = opt
        self.dataroot = opt.dataroot
        self.phase = opt.phase

        self.dir_A = os.path.join(self.dataroot, self.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(self.dataroot, self.phase + 'B')  # create a path '/path/to/data/trainB'

        # Storing Images and paths
        self.real_images = []
        self.fake_images = []
        self.original_of_fake_images = []
        self.psrns = []
        self.ssims = []

    def open_image(self, image_path: str):
        img = None
        if self.opt.isCloud:
            print("Cloud Inference is not supported")
        else:
            # img = Image.open(image_path).convert('RGB')
            img = mpimg.imread(image_path)
        return img

    def display_inference(self):
        e = len(self.real_images)
        fig, ax = plt.subplots(e, 4, figsize=(10, 2 * e))
        for i, (r_i, f_i, o_f_i, psnr, ssim) in enumerate(zip(self.real_images, self.fake_images,
                                                              self.original_of_fake_images, self.psrns, self.ssims)):
            ax[i, 0].imshow(r_i)
            ax[i, 0].set_title("A type Real Image")
            ax[i, 1].imshow(f_i)
            ax[i, 1].set_title("B type Fake Image")
            ax[i, 2].imshow(o_f_i)
            ax[i, 2].set_title("B type Real Image")
            ax[i, 3].axis("off")
            ax[i, 3].invert_yaxis()
            ax[i, 3].text(0.5, 0.5, f"PSNR: {psnr}\nSSIM: {ssim}", verticalalignment="top")

        fig.suptitle(t=f"PSNR: {sum(self.psrns) / len(self.psrns)}\n SSIM: {sum(self.ssims) / len(self.ssims)}")
        plt.show()

    def add_inference(self, image_data: dict, image_path: list):
        """Displays the test image data

        :param image_data: Image data from Model
        :param image_path: Image path from model
        """
        if 'fake_B' in image_data:
            real_i = tensor2im(image_data['real_A'])
            fake_i = tensor2im(image_data['fake_B'])
            original_of_fake_i = self.open_image(os.path.join(self.dir_B, os.path.basename(
                os.path.normpath(image_path[0]))))

            psnr = peak_signal_noise_ratio(original_of_fake_i, fake_i)
            ssim = structural_similarity(original_of_fake_i, fake_i, multichannel=True)

            self.real_images.append(real_i)
            self.fake_images.append(fake_i)
            self.original_of_fake_images.append(original_of_fake_i)
            self.psrns.append(psnr)
            self.ssims.append(ssim)
