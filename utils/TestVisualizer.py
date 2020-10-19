import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from . import tensor2im


class TestVisualizer:
    def __init__(self, opt):
        self.opt = opt
        self.dataroot = opt.dataroot
        self.phase = opt.phase

        if self.dataroot.startswith('gs://'):
            self.isCloud = True
            # create a path '/path/to/data/trainA'
            self.dir_A = os.path.join("/".join(self.dataroot.split("/")[3:]), self.phase + 'A')
            # create a path '/path/to/data/trainB'
            self.dir_B = os.path.join("/".join(self.dataroot.split("/")[3:]), self.phase + 'B')

        else:
            self.isCloud = False
            self.dir_A = os.path.join(self.dataroot, self.phase + 'A')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(self.dataroot, self.phase + 'B')  # create a path '/path/to/data/trainB'

    def open_image(self, image_path: str):
        img = None
        if self.isCloud:
            print("Cloud Inference is not supported")
        else:
            img = mpimg.imread(image_path)
        return img

    def display_inference(self, image_data: dict, image_path: list):
        """Displays the test image data

        :param image_data: Image data from Model
        :param image_path: Image path from model
        """
        if 'fake_B' in image_data:
            real0_image = tensor2im(image_data['real_A'])
            fake0_image = tensor2im(image_data['fake_B'])
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))
            ax1.imshow(real0_image)
            ax1.set_title("A type Real Image")
            ax2.imshow(fake0_image)
            ax2.set_title("B type Fake Image")
            ax3.imshow(self.open_image(os.path.join(self.dir_B, os.path.basename(os.path.normpath(image_path[0])))))
            ax3.set_title("B type Real Image")
            # plt.text(x=0.5, y=0.5, s=f"Image_path: {image_path[0]}")
            plt.show()
