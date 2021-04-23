import os

import time
import torch
from torch import distributed

from data import create_dataset
from model.CycleGan import CycleGan
from options.TrainOptions import TrainOptions
from utils.TrainStats import TrainStats
from utils.setup_cloud import setup_cloud

if __name__ == '__main__':
    opt = TrainOptions().parse()

    opt = setup_cloud(opt)


    #
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    #
    # device = torch.device("cuda" if use_cuda else "cpu")
    # if is_distributed:
    #     # Initialize the distributed environment.
    #     world_size = len(opt.hosts)
    #     os.environ['WORLD_SIZE'] = str(world_size)
    #     host_rank = opt.hosts.index(opt.current_host)
    #     os.environ['RANK'] = str(host_rank)
    #     distributed.init_process_group(backend=opt.backend, init_method="env://")

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = CycleGan(opt)
    stats = TrainStats(opt)

    # Training
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        print(f"Training {epoch}/{opt.n_epochs + opt.n_epochs_decay + 1}")

        # Training
        epoch_start_time = time.time()
        model.train()
        for i, data in enumerate(dataset):
            model.feed_input(data)
            model.optimize_parameters()

        training_end_time = time.time()
        # Training block ends

        # Evaluation
        model.eval()
        t_data = training_end_time - epoch_start_time  # Training Time
        t_comp = t_data / opt.batch_size  # Single input time

        # Save model generated images and losses
        if epoch % opt.visuals_freq == 0:
            print(f"Saving Visuals (epoch: {epoch})")
            stats.save_current_visuals(model.get_current_visuals(), f'img-{epoch}')
            stats.print_current_losses(epoch, model.get_current_losses(), t_comp, t_data)

        # Save model artifacts
        if epoch % opt.artifact_freq == 0:
            print(f'saving the model at the end of epoch {epoch}')
            model.save_networks(str(epoch))
            model.save_optimizers_and_scheduler(str(epoch))
        # Evaluation block ends

        print(f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t '
              f'Time Taken: {time.time() - epoch_start_time} sec')

        model.update_learning_rate()

    print("End of training!!!")
