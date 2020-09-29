from options.TrainOptions import TrainOptions
from data import create_dataset

import matplotlib.pyplot as plt

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    data = dataset.load_data()
    i = iter(data)
    n = next(i)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(n['A'][0].permute(1, 2, 0).numpy())

    ax[1].imshow(n['B'][0].permute(1, 2, 0).numpy())
    plt.show(block=True)
