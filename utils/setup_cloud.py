import os

import torch.cuda
import torch.distributed as dist

from utils import mkdirs


def setup_cloud(opt):
    if opt.cloud == "aws":
        is_distributed = len(opt.hosts) > 1 and opt.backend is not None
        # logger.debug("Distributed training - {}".format(is_distributed))
        use_cuda = opt.num_gpus > 0
        # logger.debug("Number of gpus available - {}".format(args.num_gpus))

        if is_distributed:
            # Initialize the distributed environment.
            world_size = len(opt.hosts)
            os.environ["WORLD_SIZE"] = str(world_size)
            host_rank = opt.hosts.index(opt.current_host)
            os.environ["RANK"] = str(host_rank)
            dist.init_process_group(backend=opt.backend, rank=host_rank, world_size=world_size)

        opt.is_distributed = is_distributed
        opt.use_cuda = use_cuda

    if opt.cloud == "colab":
        # logger.debug("Distributed training - {}".format(is_distributed))
        use_cuda = opt.num_gpus > 0 and torch.cuda.is_available()
        # logger.debug("Number of gpus available - {}".format(args.num_gpus))
        mkdirs(opt.model_dir)
        opt.is_distributed = False
        opt.use_cuda = use_cuda

    else:
        opt.is_distributed = False
        opt.use_cuda = False

    print(f"Using: {opt.cloud} Using Cuda: {opt.use_cuda}, using distributed training: {opt.is_distributed}")
    return opt
