from options.TestOptions import TestOptions
from data import create_dataset
from model.CycleGan import CycleGan
from utils.TestVisualizer import TestVisualizer


def main():
    opt = TestOptions().parse()
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.serial_batches = True
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.

    # Visualizer
    viz = TestVisualizer(opt)

    # Dataset
    dataset = create_dataset(opt)

    # setup Gan
    cycleGan = CycleGan(opt)
    cycleGan.load_networks(opt.load_model)

    x = next(iter(dataset))

    cycleGan.feed_input(x)
    cycleGan.compute_visuals()
    viz.display_inference(cycleGan.get_current_visuals(), cycleGan.get_current_image_path())


if __name__ == '__main__':
    main()