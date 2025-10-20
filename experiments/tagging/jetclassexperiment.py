import torch
import numpy as np
from torch.utils.data import DataLoader

import os, time

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from scipy.interpolate import interp1d

from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

from experiments.tagging.experiment import TaggingExperiment
from experiments.tagging.embedding import (
    dense_to_sparse_jet,
    embed_tagging_data,
)

from experiments.tagging.miniweaver.dataset import SimpleIterDataset
from experiments.tagging.miniweaver.loader import to_filelist


class JetClassTaggingExperiment(TaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.cfg.plotting.roc and not self.cfg.plotting.score
        self.class_names = [
            "ZJetsToNuNu",
            "HToBB",
            "HToCC",
            "HToGG",
            "HToWW4Q",
            "HToWW2Q1L",
            "TTBar",
            "TTBarLep",
            "WToQQ",
            "ZToQQ",
        ]
        self.num_outputs = len(self.class_names)

        if self.cfg.data.features == "fourmomenta":
            self.extra_scalars = 0
            self.cfg.data.data_config = (
                "experiments/tagging/miniweaver/configs_jetclass/fourmomenta.yaml"
            )
        elif self.cfg.data.features == "pid":
            self.extra_scalars = 6
            self.cfg.data.data_config = (
                "experiments/tagging/miniweaver/configs_jetclass/pid.yaml"
            )
        elif self.cfg.data.features == "displacements":
            self.extra_scalars = 4
            self.cfg.data.data_config = (
                "experiments/tagging/miniweaver/configs_jetclass/displacements.yaml"
            )
        elif self.cfg.data.features == "default":
            self.extra_scalars = 10
            self.cfg.data.data_config = (
                "experiments/tagging/miniweaver/configs_jetclass/default.yaml"
            )
        else:
            raise ValueError(
                f"Input feature option {self.cfg.data.features} not implemented"
            )

    def _init_loss(self):
        self.loss = torch.nn.CrossEntropyLoss()

    def init_data(self):
        LOGGER.info("Creating SimpleIterDataset")
        t0 = time.time()

        datasets = {"train": None, "test": None, "val": None}

        for_training = {"train": True, "val": True, "test": False}
        folder = {"train": "train_100M", "test": "test_20M", "val": "val_5M"}
        files_range = {
            "train": self.cfg.data.train_files_range,
            "test": self.cfg.data.test_files_range,
            "val": self.cfg.data.val_files_range,
        }
        self.num_files = {
            label: frange[1] - frange[0] for label, frange in files_range.items()
        }
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
                extra_selection=self.cfg.jc_params.extra_selection,
                remake_weights=not self.cfg.jc_params.not_remake_weights,
                load_range_and_fraction=(
                    (0, 1),
                    1,
                    self.cfg.jc_params.split_num,
                ),
                file_fraction=1,
                fetch_by_files=self.cfg.jc_params.fetch_by_files,
                fetch_step=self.cfg.jc_params.fetch_step,
                infinity_mode=self.cfg.jc_params.steps_per_epoch is not None,
                in_memory=self.cfg.jc_params.in_memory,
                name=label,
                events_per_file=self.cfg.jc_params.events_per_file,
                async_load=self.cfg.jc_params.async_load,
            )
        self.data_train = datasets["train"]
        self.data_test = datasets["test"]
        self.data_val = datasets["val"]

        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt/60:.2f} min")

    def _init_dataloader(self):
        self.loader_kwargs = {
            "pin_memory": True,
            "persistent_workers": self.cfg.jc_params.num_workers > 0
            and self.cfg.jc_params.steps_per_epoch is not None,
        }
        num_workers = {
            label: min(self.cfg.jc_params.num_workers, self.num_files[label])
            for label in ["train", "test", "val"]
        }

        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize // self.world_size,
            drop_last=True,
            num_workers=num_workers["train"],
            **self.loader_kwargs,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            drop_last=True,
            num_workers=num_workers["val"],
            **self.loader_kwargs,
        )
        self.test_loader = DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            drop_last=False,
            num_workers=num_workers["test"],
            **self.loader_kwargs,
        )

        self.init_standardization()

    @torch.no_grad()
    def _evaluate_single(self, loader, title, mode, step=None):
        assert mode in ["val", "eval"]

        if mode == "eval":
            LOGGER.info(f"### Starting to evaluate model on {title} dataset ###")
        metrics = {}

        # predictions
        labels_true, labels_predict = [], []
        self.model.eval()
        for batch in loader:
            y_pred, label, _, _ = self._get_ypred_and_label(batch)
            labels_true.append(label.cpu())
            labels_predict.append(y_pred.cpu().float())

        labels_true, labels_predict = torch.cat(labels_true), torch.cat(labels_predict)
        if mode == "eval":
            metrics["labels_true"], metrics["labels_predict"] = (
                labels_true,
                labels_predict,
            )

        # ce loss
        metrics["loss"] = torch.nn.functional.cross_entropy(
            labels_predict, labels_true
        ).item()
        labels_true, labels_predict = (
            labels_true.numpy(),
            torch.softmax(labels_predict, dim=1).numpy(),
        )

        # accuracy
        metrics["accuracy"] = accuracy_score(labels_true, labels_predict.argmax(1))
        if mode == "eval":
            LOGGER.info(f"Accuracy on {title} dataset:\t{metrics['accuracy']:.4f}")

        # auc and roc (fpr = epsB, tpr = epsS)
        metrics["auc_ovo"] = roc_auc_score(
            labels_true, labels_predict, multi_class="ovo", average="macro"
        )  # unweighted mean of AUCs across classes
        if mode == "eval":
            LOGGER.info(f"The ovo mean AUC is\t\t{metrics['auc_ovo']:.5f}")

        # 1/epsB at fixed epsS
        def get_rej(epsS, tpr, fpr):
            background_eff_fn = interp1d(tpr, fpr)
            return 1 / background_eff_fn(epsS)

        class_rej_dict = [None, 0.5, 0.5, 0.5, 0.5, 0.99, 0.5, 0.995, 0.5, 0.5]

        for i in range(1, len(self.class_names)):
            labels_predict_class = labels_predict[
                (labels_true == 0) | (labels_true == i)
            ]
            labels_true_class = labels_true[(labels_true == 0) | (labels_true == i)]
            labels_predict_class = labels_predict_class[:, [0, i]]

            denom = labels_predict_class[:, 0] + labels_predict_class[:, 1]
            predict_score = labels_predict_class[:, 1] / np.clip(
                denom, a_min=1e-10, a_max=None
            )

            fpr, tpr, _ = roc_curve(labels_true_class == i, predict_score)

            rej_string = str(class_rej_dict[i]).replace(".", "")
            metrics[f"rej{rej_string}_{i}"] = get_rej(class_rej_dict[i], tpr, fpr)
            if mode == "eval":
                LOGGER.info(
                    f"Rejection rate for class {self.class_names[i]:>10} on {title} dataset:{metrics[f'rej{rej_string}_{i}']:>5.0f} (epsS={class_rej_dict[i]})"
                )

        # create latex string
        if mode == "eval":
            tex_string = f"{self.cfg.run_name} & {metrics['accuracy']:.3f} & {metrics['auc_ovo']:.3f}"
            for i, rej in enumerate(class_rej_dict):
                if rej is None:
                    continue
                rej_string = str(rej).replace(".", "")
                tex_string += f" & {metrics[f'rej{rej_string}_{i}']:.0f}"
            tex_string += r" \\"
            LOGGER.info(tex_string)

        if self.cfg.use_mlflow:
            for key, value in metrics.items():
                if key in ["labels_true", "labels_predict"]:
                    # do not log matrices
                    continue
                name = f"{mode}.{title}" if mode == "eval" else "val"
                log_mlflow(f"{name}.{key}", value, step=step)

        return metrics

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
        label = batch[1]["_label_"].to(self.device).to(torch.long)
        fourmomenta, scalars, ptr = dense_to_sparse_jet(fourmomenta, scalars)
        return fourmomenta, scalars, ptr, label
