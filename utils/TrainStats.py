import os
import time
import torch
import numpy as np
from PIL import Image
from google.cloud import storage


class TrainStats:
    def __init__(self, opt):

        # create a logging file to store training losses
        self.isCloud = opt.checkpoints_dir.startswith('gs://')
        if self.isCloud:
            # G-Cloud
            self.bucket = self.setup_cloud_bucket(opt.checkpoints_dir)
            self.log_name = 'loss_log.txt'
            self.cloud_log_name = os.path.join("/".join(opt.checkpoints_dir.split("/")[3:]), opt.name, 'loss_log.txt')
            self.img_dir = os.path.join("/".join(opt.checkpoints_dir.split("/")[3:]), opt.name, 'visuals')
        else:
            # Local
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'visuals')

        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def setup_cloud_bucket(self, dataroot):
        """Setup Google Cloud Bucket

        :type dataroot: str
        :param dataroot: The Root of the Data-storage
        :return: Bucket
        """
        bucket_name = dataroot.split("/")[2]
        print(f"Using Bucket: {bucket_name} for storing artifacts")
        c = storage.Client()
        b = c.get_bucket(bucket_name)

        assert b.exists(), f"Bucket {bucket_name} dos't exist. Try different one"

        return b

    def save_file_to_cloud(self, file_path_cloud, file_path_local):
        self.bucket.blob(file_path_cloud).upload_from_filename(file_path_local)

    def save_loss_to_cloud(self,):
        if self.isCloud:
            self.save_file_to_cloud(self.cloud_log_name, self.log_name)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        :param epoch: current epoch
        :type epoch int
        :param iters: current training iteration during this epoch (reset to 0 at the end of every epoch)
        :type iters int
        :param losses: training losses stored in the format of (name, float) pairs
        :type losses dict
        :param t_comp: computational time per data point (normalized by batch_size)
        :type t_comp float
        :param t_data: data loading time per data point (normalized by batch_size)
        :type t_data float
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def save_current_visuals(self, images, prefix):
        """Save Current Produced images

        :param images: Images
        :type images dict
        :param prefix: Name prefix
        :return:
        """
        for label, image in images.items():
            imageName = '%s-%s.jpg' % (prefix, label)
            img = self.tensor2im(image)
            self.save_image(img, imageName)

    def tensor2im(self, input_image, imtype=np.uint8):
        """Converts a Tensor array into a numpy image array.

        :param input_image: the input image tensor array
        :param imtype: the desired type of the converted numpy array
        :return: Converted Image
        """
        if not isinstance(input_image, np.ndarray):
            if isinstance(input_image, torch.Tensor):  # get the data from a variable
                image_tensor = input_image.data
            else:
                return input_image
            image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))

            # post-processing: transpose and scaling
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        else:  # if it is a numpy array, do nothing
            image_numpy = input_image
        return image_numpy.astype(imtype)

    def save_image(self, image_numpy, image_name, aspect_ratio=1.0):
        """Save a numpy image to the disk

        :param image_numpy: input numpy array
        :param image_name: the path of the image
        :param aspect_ratio: aspect_ratio of the image
        """
        if self.isCloud:
            image_path = image_name
        else:
            image_path = os.path.join(self.img_dir, image_name)
        image_pil = Image.fromarray(image_numpy)
        h, w, _ = image_numpy.shape

        if aspect_ratio > 1.0:
            image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
        if aspect_ratio < 1.0:
            image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
        image_pil.save(image_path)

        if self.isCloud:
            self.save_file_to_cloud(os.path.join(self.img_dir, image_path), image_path)
            os.remove(image_path)
