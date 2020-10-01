from options.TrainOptions import TrainOptions
from data import create_dataset
from model.CycleGan import CycleGan

import matplotlib.pyplot as plt

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = CycleGan(opt)

    i = iter(dataset)
    n = next(i)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(n['A'][0].permute(1, 2, 0).numpy())

    ax[1].imshow(n['B'][0].permute(1, 2, 0).numpy())
    plt.show(block=True)
