import time
from data import create_dataset
from model.CycleGan import CycleGan
from options.TrainOptions import TrainOptions
from utils.TrainStats import TrainStats

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    stats = TrainStats(opt)

    model = CycleGan(opt)
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        for i, data in enumerate(dataset):
            model.feed_input(data)
            iter_start_time = time.time()
            model.optimize_parameters()

            total_iters += 1
            epoch_iter += 1

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - epoch_start_time
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                stats.print_current_losses(epoch, epoch_iter, model.get_current_losses(), t_comp, t_data)

            if total_iters % opt.display_freq == 0:
                print(f"Saving Visuals (epoch: {epoch}, total_iters: {total_iters})")
                stats.save_current_visuals(model.get_current_visuals(), 'img-%s' % total_iters)

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                model.save_optimizers(save_suffix)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_optimizers('latest')
            model.save_networks(str(epoch))
            model.save_optimizers(str(epoch))

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

    print("End of training!!!")
