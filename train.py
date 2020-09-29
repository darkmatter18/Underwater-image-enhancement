from options.TrainOptions import TrainOptions

if __name__ == '__main__':
    opt = TrainOptions().parse()
    print(opt.crop_size)
