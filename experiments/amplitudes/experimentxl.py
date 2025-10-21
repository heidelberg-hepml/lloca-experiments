import os
import queue
import threading
import random
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from experiments.amplitudes.utils import (
    load_file,
)
from experiments.logger import LOGGER
from experiments.amplitudes.experiment import AmplitudeExperiment


class AmplitudeXLExperiment(AmplitudeExperiment):
    def init_data(self):
        assert not "train" in self.cfg.evaluation.eval_set

        real_subsample = self.cfg.data.subsample
        self.cfg.data.subsample = None
        super().init_data()
        self.cfg.data.subsample = real_subsample

    def _init_dataloader(self):
        super()._init_dataloader(log=False)  # init val and test dataloaders

        assert (
            self.cfg.data.subsample is None or self.cfg.data.num_train_files == 1
        ), "You should not subsample while using multiple files"

        # overwrite self.train_loader
        get_fname = lambda n: os.path.join(
            self.cfg.data.data_path, f"{self.dataset}_{n}.npy"
        )
        file_paths = [get_fname(n + 1) for n in range(self.cfg.data.num_train_files)]
        loading_kwargs = dict(
            cfg_data=self.cfg.data,
            dataset=self.dataset,
            amp_mean=self.amp_mean,
            amp_std=self.amp_std,
            mom_std=self.mom_std,
            network_float64=self.cfg.use_float64,
            momentum_float64=self.cfg.data.momentum_float64,
        )
        trn_set = PrefetchFilesDataset(
            file_paths,
            num_prefetch=self.cfg.data.num_prefetch,
            **loading_kwargs,
        )
        self.train_loader = torch.utils.data.DataLoader(
            dataset=trn_set,
            batch_size=self.cfg.training.batchsize,
            num_workers=0,
        )

        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )


class PrefetchFilesDataset(IterableDataset):
    """
    Custom dataset to load files on the fly with this strategy:
    - Prepare num_prefetch files in a seperate thread using the load_file function.
    - The DataLoader can access the prepared events with the __iter__ function.
    - We shuffle randomly between files and between events in a single file.
    - Note: cfg_data.subsample allows to control how many events are loaded from each file.
    """

    def __init__(
        self,
        file_paths,
        num_prefetch=2,
        events_per_file=1000000,
        **loading_kwargs,
    ):
        """
        Parameters
        ----------
        file_paths : list of str
            List of paths to the files to load.
        num_prefetch : int
            Number of files to prefetch.
        events_per_file : int
            Number of events to yield from each file.
        **loading_kwargs
        """
        super().__init__()
        subsample = loading_kwargs["cfg_data"].subsample
        self.events_per_file = events_per_file if subsample is None else subsample

        # prefetch params
        self.file_paths = list(file_paths)
        self.num_prefetch = num_prefetch
        self.epoch = 0
        self.base_seed = 42
        self.rng = random.Random(self.base_seed)
        self._EOF = object()

        self.loading_kwargs = loading_kwargs

    def __len__(self):
        # global length of dataset (across processes)
        return len(self.file_paths) * self.events_per_file

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _worker(self, file_queue, files_for_me):
        for local_i, fpath in enumerate(files_for_me):
            # always use the same initial randomness for each file
            g = torch.Generator().manual_seed(
                (self.base_seed + self.epoch) ^ (local_i + 0x9E3779B97F4A7C15)
            )
            amp, mom, _, _, _ = load_file(fpath, generator=g, **self.loading_kwargs)
            n = min(self.events_per_file, amp.shape[0])
            idx = torch.randperm(amp.shape[0], generator=g)[:n]
            amp, mom = amp[idx], mom[idx]
            file_queue.put((amp, mom))

        file_queue.put(self._EOF)

    def __iter__(self):
        files_for_me = self._files_for_this_consumer()

        file_queue = queue.Queue(maxsize=self.num_prefetch)
        worker = threading.Thread(target=self._worker, args=(file_queue, files_for_me))
        worker.daemon = True  # exit if main thread exits
        worker.start()

        while True:
            data = file_queue.get()
            if data is self._EOF:
                break

            amp, mom = data
            for i in range(amp.shape[0]):
                yield amp[i], mom[i]

        worker.join()  # terminate

    def _parallel_context(self):
        rank, world_size = 0, 1
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        wi = get_worker_info()
        worker_id, num_workers = (wi.id, wi.num_workers) if wi is not None else (0, 1)

        consumers = world_size * num_workers
        consumer_id = rank * num_workers + worker_id
        return consumer_id, consumers

    def _files_for_this_consumer(self):
        # reshuffle the file list each epoch with a deterministic seed
        seed = self.base_seed + self.epoch
        rng = random.Random(seed)
        shuffled = list(self.file_paths)
        rng.shuffle(shuffled)

        consumer_id, consumers = self._parallel_context()
        # shard by file index for locality
        return [f for i, f in enumerate(shuffled) if (i % consumers) == consumer_id]
