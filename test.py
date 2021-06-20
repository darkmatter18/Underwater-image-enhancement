from options.TestOptions import TestOptions
from data import create_dataset
from model import create_model
from utils.TestVisualizer import TestVisualizer


def main():
    opt = TestOptions().parse()
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    # disable data shuffling; comment this line if results on randomly chosen images are needed.
    # opt.serial_batches = True
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.

    # Visualizer
    viz = TestVisualizer(opt)

    # Dataset
    dataset = create_dataset(dataroot=opt.training_data_dir, serial_batches=opt.serial_batches,
                             preprocess=opt.preprocess, no_flip=opt.no_flip, load_size=opt.load_size,
                             crop_size=opt.crop_size, batch_size=opt.batch_size, num_threads=opt.num_threads,
                             is_distributed=opt.is_distributed, use_cuda=opt.use_cuda)

    # setup Gan
    cycleGan = create_model(opt)
    cycleGan.load_networks(opt.load_model)

    dataset_iter = iter(dataset)
    no_of_examples = int(opt.examples)

    print(f"Testing for {no_of_examples} examples")
    while no_of_examples > 0:
        data = next(dataset_iter)
        cycleGan.feed_input(data)
        cycleGan.compute_visuals()
        viz.add_inference(cycleGan.get_current_visuals(), cycleGan.get_current_image_path())
        no_of_examples -= 1

    viz.display_inference()


if __name__ == '__main__':
    main()
