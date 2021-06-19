import os
import torch.distributed as dist


def setup_cloud(opt):
    if opt.cloud == "aws":
        is_distributed = len(opt.hosts) > 1 and opt.backend is not None
        # logger.debug("Distributed training - {}".format(is_distributed))
        use_cuda = opt.num_gpus > 0
        # logger.debug("Number of gpus available - {}".format(args.num_gpus))
        num_workers = 1 if use_cuda else 0
        pin_memory = use_cuda
        # device = torch.device("cuda" if use_cuda else "cpu")

        if is_distributed:
            # Initialize the distributed environment.
            world_size = len(opt.hosts)
            os.environ["WORLD_SIZE"] = str(world_size)
            host_rank = opt.hosts.index(opt.current_host)
            os.environ["RANK"] = str(host_rank)
            dist.init_process_group(backend=opt.backend, rank=host_rank, world_size=world_size)

        opt.is_distributed = is_distributed
        opt.use_cuda = use_cuda
        opt.num_workers = num_workers
        opt.pin_memory = pin_memory
        return opt

    else:
        opt.is_distributed = False
        opt.use_cuda = False
        opt.num_workers = 0
        opt.pin_memory = False
        return opt
