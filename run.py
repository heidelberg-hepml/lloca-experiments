import hydra
import os
import datetime
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from experiments.tagging.experiment import TopTaggingExperiment
from experiments.tagging.finetuneexperiment import TopTaggingFineTuneExperiment
from experiments.tagging.topxlexperiment import TopXLTaggingExperiment
from experiments.tagging.jetclassexperiment import JetClassTaggingExperiment
from experiments.amplitudes.experiment import AmplitudeExperiment
from experiments.amplitudes.experimentxl import AmplitudeXLExperiment
from experiments.eventgen.processes import ttbarExperiment


@hydra.main(config_path="config_quick", config_name="toptagging", version_base=None)
def main(cfg):
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if world_size > 1:
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("NCCL_DEBUG", "WARN")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")

        _set_common_env(world_size)
        mp.spawn(ddp_worker, nprocs=world_size, args=(cfg,), join=True, daemon=False)
    else:
        # no GPU or only one GPU -> run on main process
        ddp_worker(rank=0, cfg=cfg)


def ddp_worker(rank, cfg):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    if world_size > 1:
        # set up communication between processes
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=30),
        )
        torch.cuda.set_device(rank)

    if cfg.exp_type == "toptagging":
        constructor = TopTaggingExperiment
    elif cfg.exp_type == "toptaggingft":
        constructor = TopTaggingFineTuneExperiment
    elif cfg.exp_type == "topxltagging":
        constructor = TopXLTaggingExperiment
    elif cfg.exp_type == "jctagging":
        constructor = JetClassTaggingExperiment
    elif cfg.exp_type == "amplitudes":
        constructor = AmplitudeExperiment
    elif cfg.exp_type == "amplitudesxl":
        constructor = AmplitudeXLExperiment
    elif cfg.exp_type == "ttbar":
        constructor = ttbarExperiment
    else:
        raise ValueError(f"exp_type {cfg.exp_type} not implemented")

    exp = constructor(cfg, rank, world_size)
    exp()

    if world_size > 1:
        dist.barrier(device_ids=[rank])
        dist.destroy_process_group()


def _set_common_env(world_size):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
    os.environ["WORLD_SIZE"] = str(world_size)


def _find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    main()
