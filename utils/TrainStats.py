import os
import time
import pickle
from . import tensor2im
from PIL import Image


class TrainStats:
    def __init__(self, opt):
        # create a logging file to store training losses
        self.log_loss_dir = opt.output_data_dir
        self.img_dir = os.path.join(opt.output_data_dir, 'visuals')
        self.log_file_name = f'loss_log_{str(int(time.time()))}.txt'
        self.loss_file_name = f'loss_stats_{str(int(time.time()))}.pkl'
        self.losses = {'loss_idt_A': [], 'loss_idt_B': [], 'loss_D_A': [], 'loss_D_B': [], 'loss_G_AtoB': [],
                       'loss_G_BtoA': [], 'cycle_loss_A': [], 'cycle_loss_B': []}

        lss = os.path.join(self.log_loss_dir, self.log_file_name)

        with open(lss, "w+") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def print_current_losses(self, epoch, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        :param epoch: current epoch
        :type epoch int
        :param losses: training losses stored in the format of (name, float) pairs
        :type losses dict
        :param t_comp: computational time per data point (normalized by batch_size)
        :type t_comp float
        :param t_data: data loading time per data point (normalized by batch_size)
        :type t_data float
        """
        message = '(epoch: %d, time: %.3f, data: %.3f) ' % (epoch, t_comp, t_data)
        for k, v in losses.items():
            self.losses[k].append(v)
            message += '%s: %.3f ' % (k, v)

        lss = os.path.join(self.log_loss_dir, self.log_file_name)
        liss = os.path.join(self.log_loss_dir, self.loss_file_name)

        print(message)  # print the message
        with open(lss, "a") as log_file:
            log_file.write('%s\n' % message)

        with open(liss, 'wb') as f:
            pickle.dump(self.losses, f)

    def save_current_visuals(self, images, prefix):
        """Save Current Produced images

        :param images: Images
        :type images dict
        :param prefix: Name prefix
        :return:
        """
        for label, image in images.items():
            imageName = '%s-%s.jpg' % (prefix, label)
            img = tensor2im(image)
            self.save_image(img, imageName)

    def save_image(self, image_numpy, image_name, aspect_ratio=1.0):
        """Save a numpy image to the disk

        :param image_numpy: input numpy array
        :param image_name: the path of the image
        :param aspect_ratio: aspect_ratio of the image
        """

        image_path = os.path.join(self.img_dir, image_name)
        image_pil = Image.fromarray(image_numpy)
        h, w, _ = image_numpy.shape

        if aspect_ratio > 1.0:
            image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
        if aspect_ratio < 1.0:
            image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
        image_pil.save(image_path)
