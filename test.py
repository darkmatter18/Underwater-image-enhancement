from options.TestOptions import TestOptions
from data import create_dataset
from model import create_model
from utils.TestVisualizer import TestVisualizer
from utils.setup_cloud import setup_cloud


def main():
    opt = TestOptions().parse()
    opt = setup_cloud(opt)
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.

    # Visualizer
    viz = TestVisualizer(opt)

    # Dataset
    dataset = create_dataset(dataroot=opt.test_dataset_dir, subdir=opt.test_subdir, phase=opt.phase,
                             serial_batches=opt.serial_batches, preprocess=opt.preprocess, no_flip=opt.no_flip,
                             load_size=opt.load_size, crop_size=opt.crop_size, batch_size=opt.batch_size,
                             is_distributed=opt.is_distributed, use_cuda=opt.use_cuda, is_test=True)

    # setup Gan
    cycleGan = create_model(opt)
    cycleGan.load_networks(opt.load_model)

    if opt.all:
        for data in dataset:
            cycleGan.feed_input(data)
            cycleGan.compute_visuals()
            viz.add_inference(cycleGan.get_current_visuals(), cycleGan.get_current_image_path())
    else:
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
