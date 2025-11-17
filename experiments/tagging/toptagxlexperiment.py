import os
import time

import torch
from torch.utils.data import DataLoader

from experiments.logger import LOGGER
from experiments.tagging.embedding import dense_to_sparse_jet
from experiments.tagging.experiment import TaggingExperiment
from experiments.tagging.miniweaver.dataset import SimpleIterDataset
from experiments.tagging.miniweaver.loader import to_filelist


class TopTagXLExperiment(TaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = 1
        self.class_names = ["qcd", "top"]

        if self.cfg.data.features == "fourmomenta":
            self.extra_scalars = 0
            self.cfg.data.data_config = (
                "experiments/tagging/miniweaver/configs_topxl/fourmomenta.yaml"
            )
        elif self.cfg.data.features == "pid":
            self.extra_scalars = 6
            self.cfg.data.data_config = "experiments/tagging/miniweaver/configs_topxl/pid.yaml"
        elif self.cfg.data.features == "displacements":
            self.extra_scalars = 4
            self.cfg.data.data_config = (
                "experiments/tagging/miniweaver/configs_topxl/displacements.yaml"
            )
        elif self.cfg.data.features == "default":
            self.extra_scalars = 10
            self.cfg.data.data_config = "experiments/tagging/miniweaver/configs_topxl/default.yaml"
        else:
            raise ValueError(f"Input feature option {self.cfg.data.features} not implemented")

    def init_data(self):
        LOGGER.info("Creating SimpleIterDataset")
        t0 = time.time()

        datasets = {"train": None, "test": None, "val": None}

        for_training = {"train": True, "val": True, "test": False}
        folder = {"train": "train_100M", "test": "test_25M", "val": "val_10M"}
        files_range = {
            "train": self.cfg.data.train_files_range,
            "test": self.cfg.data.test_files_range,
            "val": self.cfg.data.val_files_range,
        }
        self.num_files = {label: frange[1] - frange[0] for label, frange in files_range.items()}
        for label in ["train", "test", "val"]:
            path = os.path.join(self.cfg.data.data_dir, folder[label])
            flist = [
                f"{classname}:{path}/{classname}_{str(i).zfill(3)}.root"
                for classname in self.class_names
                for i in range(*files_range[label])
            ]
            file_dict, _ = to_filelist(flist)

            LOGGER.info(f"Using {len(flist)} files for {label}ing from {path}")
            datasets[label] = SimpleIterDataset(
                file_dict,
                self.cfg.data.data_config,
                for_training=for_training[label],
                extra_selection=self.cfg.topxl_params.extra_selection,
                remake_weights=not self.cfg.topxl_params.not_remake_weights,
                load_range_and_fraction=((0, 1), 1, 1),
                file_fraction=1,
                fetch_by_files=self.cfg.topxl_params.fetch_by_files,
                fetch_step=self.cfg.topxl_params.fetch_step,
                infinity_mode=self.cfg.topxl_params.steps_per_epoch is not None,
                in_memory=self.cfg.topxl_params.in_memory,
                name=label,
                events_per_file=self.cfg.topxl_params.events_per_file,
                async_load=self.cfg.topxl_params.async_load,
            )
        self.data_train = datasets["train"]
        self.data_test = datasets["test"]
        self.data_val = datasets["val"]

        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt / 60:.2f} min")

    def _init_dataloader(self):
        self.loader_kwargs = {
            "pin_memory": True,
            "persistent_workers": self.cfg.topxl_params.num_workers > 0
            and self.cfg.topxl_params.steps_per_epoch is not None,
        }
        num_workers = {
            label: min(self.cfg.topxl_params.num_workers, self.num_files[label])
            for label in ["train", "test", "val"]
        }

        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize // self.world_size,
            drop_last=True,
            num_workers=num_workers["train"],
            multiprocessing_context="fork",
            **self.loader_kwargs,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            drop_last=True,
            num_workers=num_workers["val"],
            multiprocessing_context="fork",
            **self.loader_kwargs,
        )
        self.test_loader = DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            drop_last=False,
            num_workers=num_workers["test"],
            multiprocessing_context="fork",
            **self.loader_kwargs,
        )

        self.init_standardization()

    def _extract_batch(self, batch):
        fourmomenta = batch[0]["pf_vectors"].to(self.device, self.momentum_dtype)
        if self.cfg.data.features == "fourmomenta":
            scalars = torch.empty(
                fourmomenta.shape[0],
                0,
                fourmomenta.shape[2],
                device=fourmomenta.device,
                dtype=self.dtype,
            )
        else:
            scalars = batch[0]["pf_features"].to(self.device, self.dtype)
        label = batch[1]["_label_"].to(self.device)
        fourmomenta, scalars, ptr = dense_to_sparse_jet(fourmomenta, scalars)
        label = label.to(self.dtype)
        return fourmomenta, scalars, ptr, label
