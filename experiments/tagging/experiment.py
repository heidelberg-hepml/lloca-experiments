import numpy as np
import torch
from torch_geometric.loader import DataLoader
import os, time

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

from experiments.base_experiment import BaseExperiment
from experiments.tagging.dataset import TopTaggingDataset
from experiments.tagging.embedding import embed_tagging_data, get_num_tagging_features
from experiments.tagging.plots import plot_mixer
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow


class TaggingExperiment(BaseExperiment):
    """
    Base class for jet tagging experiments, focusing on binary classification
    """

    def init_physics(self):
        modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
        self.momentum_dtype = (
            torch.float64 if self.cfg.data.momentum_float64 else torch.float32
        )

        self.cfg.model.out_channels = self.num_outputs
        if modelname == "LGATr":
            self.cfg.model.net.in_s_channels = (
                0 if self.cfg.model.mean_aggregation else 1
            )
            self.cfg.model.net.in_s_channels += self.extra_scalars
        elif modelname == "LorentzNet":
            self.cfg.model.net.n_scalar = self.extra_scalars
        elif modelname == "PELICAN":
            self.cfg.model.net.num_scalars = self.extra_scalars
        elif modelname == "CGENN":
            # CGENN cant handle zero scalar inputs -> give 1 input with zeros
            self.cfg.model.net.in_features_h = 1 + self.extra_scalars
        elif modelname in [
            "Transformer",
            "ParticleTransformer",
            "GraphNet",
            "ParticleNet",
            "MIParticleTransformer",
        ]:
            # LLoCa models
            num_tagging_features = get_num_tagging_features(
                only_ztransform=self.cfg.data.only_ztransform_tagging_features
            )
            self.cfg.model.only_ztransform_tagging_features = (
                self.cfg.data.only_ztransform_tagging_features
            )
            self.cfg.model.in_channels = num_tagging_features + self.extra_scalars
            if self.cfg.model.add_fourmomenta_backbone:
                self.cfg.model.in_channels += 4

            if modelname == "Transformer":
                self.cfg.model.in_channels += (
                    0 if self.cfg.model.mean_aggregation else 1
                )
            elif modelname == "GraphNet":
                self.cfg.model.net.num_edge_attr = (
                    1 if self.cfg.model.include_edges else 0
                )
            elif modelname == "ParticleNet":
                self.cfg.model.net.hidden_reps_list[
                    0
                ] = f"{self.cfg.model.in_channels}x0n"

            # decide which entries to use for the framesnet
            if "equivectors" in self.cfg.model.framesnet:
                self.cfg.model.framesnet.equivectors.num_scalars = self.extra_scalars
                self.cfg.model.framesnet.equivectors.num_scalars += (
                    num_tagging_features
                    if self.cfg.data.add_tagging_features_framesnet
                    else 0
                )
        else:
            raise NotImplementedError(f"Model {modelname} not implemented")

    def init_data(self):
        raise NotImplementedError

    def _init_data(self, Dataset, data_path):
        LOGGER.info(f"Creating {Dataset.__name__} from {data_path}")
        t0 = time.time()
        self.data_train = Dataset()
        self.data_test = Dataset()
        self.data_val = Dataset()
        kwargs = dict(
            network_float64=self.cfg.use_float64,
            momentum_float64=self.cfg.data.momentum_float64,
        )
        self.data_train.load_data(data_path, "train", **kwargs)
        self.data_test.load_data(data_path, "test", **kwargs)
        self.data_val.load_data(data_path, "val", **kwargs)
        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt/60:.2f} min")

    def _init_dataloader(self):
        trn_sampler = torch.utils.data.DistributedSampler(
            self.data_train,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        tst_sampler = torch.utils.data.DistributedSampler(
            self.data_test,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )
        val_sampler = torch.utils.data.DistributedSampler(
            self.data_val,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )

        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize // self.world_size,
            sampler=trn_sampler,
        )
        self.test_loader = DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            sampler=tst_sampler,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            sampler=val_sampler,
        )

        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

        self.init_standardization()

    def init_standardization(self):
        if hasattr(self.model, "init_standardization"):
            batch = next(iter(self.train_loader))
            fourmomenta, scalars, ptr, _ = self._extract_batch(batch)
            embedding = embed_tagging_data(
                fourmomenta,
                scalars,
                ptr,
                self.cfg.data,
            )
            self.model.init_standardization(embedding["fourmomenta"], embedding["ptr"])

    def _init_optimizer(self, param_groups=None):
        if self.cfg.model.net._target_.rsplit(".", 1)[-1] in [
            "ParticleTransformer",
            "MIParticleTransformer",
        ]:
            # special treatment for ParT, see
            # https://github.com/hqucms/weaver-core/blob/dev/custom_train_eval/weaver/train.py#L464
            decay, no_decay = {}, {}
            for name, param in self.model.net.named_parameters():
                if not param.requires_grad:
                    continue
                if (
                    len(param.shape) == 1
                    or name.endswith(".bias")
                    or (
                        hasattr(self.model.net, "no_weight_decay")
                        and name in {"cls_token"}
                    )
                ):
                    no_decay[name] = param
                else:
                    decay[name] = param
            decay_1x, no_decay_1x = list(decay.values()), list(no_decay.values())
            param_groups = [
                {
                    "params": no_decay_1x,
                    "weight_decay": 0.0,
                    "lr": self.cfg.training.lr,
                },
                {
                    "params": decay_1x,
                    "weight_decay": self.cfg.training.weight_decay,
                    "lr": self.cfg.training.lr,
                },
                {
                    "params": self.model.framesnet.parameters(),
                    "weight_decay": self.cfg.training.weight_decay_framesnet,
                    "lr": self.cfg.training.lr * self.cfg.training.lr_factor_framesnet,
                },
            ]

        super()._init_optimizer(param_groups=param_groups)

    def evaluate(self):
        self.results = {}
        loader_dict = {
            "train": self.train_loader,
            "test": self.test_loader,
            "val": self.val_loader,
        }
        for set_label in self.cfg.evaluation.eval_set:
            if self.ema is not None:
                with self.ema.average_parameters():
                    self.results[set_label] = self._evaluate_single(
                        loader_dict[set_label], f"{set_label}_ema", mode="eval"
                    )

                self._evaluate_single(loader_dict[set_label], set_label, mode="eval")

            else:
                self.results[set_label] = self._evaluate_single(
                    loader_dict[set_label], set_label, mode="eval"
                )

    @torch.no_grad()
    def _evaluate_single(self, loader, title, mode, step=None):
        assert mode in ["val", "eval"]

        if mode == "eval":
            LOGGER.info(
                f"### Starting to evaluate model on {title} dataset with "
                f"{len(loader.dataset)} elements, batchsize {loader.batch_size} ###"
            )
        metrics = {}

        # predictions
        labels_true, labels_predict = [], []
        self.model.eval()
        for batch in loader:
            y_pred, label, _, _ = self._get_ypred_and_label(batch)
            y_pred = torch.nn.functional.sigmoid(y_pred)
            labels_true.append(label.cpu().float())
            labels_predict.append(y_pred.cpu().float())
        labels_true, labels_predict = torch.cat(labels_true), torch.cat(labels_predict)

        if mode == "eval":
            metrics["labels_true"], metrics["labels_predict"] = (
                labels_true,
                labels_predict,
            )

        # bce loss
        metrics["loss"] = torch.nn.functional.binary_cross_entropy(
            labels_predict, labels_true
        ).item()
        labels_true, labels_predict = labels_true.numpy(), labels_predict.numpy()

        # accuracy
        metrics["accuracy"] = accuracy_score(labels_true, np.round(labels_predict))
        if mode == "eval":
            LOGGER.info(f"Accuracy on {title} dataset: {metrics['accuracy']:.4f}")

        # roc (fpr = epsB, tpr = epsS)
        fpr, tpr, th = roc_curve(labels_true, labels_predict)
        if mode == "eval":
            metrics["fpr"], metrics["tpr"] = fpr, tpr
        metrics["auc"] = roc_auc_score(labels_true, labels_predict)
        if mode == "eval":
            LOGGER.info(f"AUC score on {title} dataset: {metrics['auc']:.4f}")

        # 1/epsB at fixed epsS
        def get_rej(epsS):
            idx = np.argmin(np.abs(tpr - epsS))
            return 1 / fpr[idx]

        metrics["rej03"] = get_rej(0.3)
        metrics["rej05"] = get_rej(0.5)
        metrics["rej08"] = get_rej(0.8)
        if mode == "eval":
            LOGGER.info(
                f"Rejection rate {title} dataset: {metrics['rej03']:.0f} (epsS=0.3), "
                f"{metrics['rej05']:.0f} (epsS=0.5), {metrics['rej08']:.0f} (epsS=0.8)"
            )

        if self.cfg.use_mlflow:
            for key, value in metrics.items():
                if key in ["labels_true", "labels_predict", "fpr", "tpr"]:
                    # do not log matrices
                    continue
                name = f"{mode}.{title}" if mode == "eval" else "val"
                log_mlflow(f"{name}.{key}", value, step=step)

        if mode == "eval":
            framesString = type(self.model.framesnet).__name__
            num_parameters = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            LOGGER.info(
                f"table {title}: {framesString} ({self.cfg.training.iterations} iterations)"
                f" & {num_parameters} & {metrics['accuracy']:.4f} & {metrics['auc']:.4f}"
                f" & {metrics['rej03']:.0f} & {metrics['rej05']:.0f} & {metrics['rej08']:.0f} \\\\"
            )
        return metrics

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path, exist_ok=True)
        title = type(self.model.net).__name__
        LOGGER.info(f"Creating plots in {plot_path}")

        if (
            self.cfg.evaluation.save_roc
            and self.cfg.evaluate
            and ("test" in self.cfg.evaluation.eval_set)
        ):
            file = f"{plot_path}/roc.txt"
            roc = np.stack(
                (self.results["test"]["fpr"], self.results["test"]["tpr"]), axis=-1
            )
            np.savetxt(file, roc)

        plot_dict = {}
        if self.cfg.evaluate and ("test" in self.cfg.evaluation.eval_set):
            plot_dict = {"results_test": self.results["test"]}
        if self.cfg.train:
            plot_dict["train_loss"] = self.train_loss
            plot_dict["val_loss"] = self.val_loss
            plot_dict["train_lr"] = self.train_lr
            plot_dict["grad_norm"] = torch.stack(self.grad_norm_train).cpu()
            plot_dict["grad_norm_frames"] = torch.stack(self.grad_norm_frames).cpu()
            plot_dict["grad_norm_net"] = torch.stack(self.grad_norm_net).cpu()
            for key, value in self.train_metrics.items():
                plot_dict[key] = value
        plot_mixer(self.cfg, plot_path, title, plot_dict)

    def _init_loss(self):
        self.loss = torch.nn.BCEWithLogitsLoss()

    # overwrite _validate method to compute metrics over the full validation set
    def _validate(self, step):
        if self.ema is not None:
            with self.ema.average_parameters():
                metrics = self._evaluate_single(
                    self.val_loader, "val", mode="val", step=step
                )
        else:
            metrics = self._evaluate_single(
                self.val_loader, "val", mode="val", step=step
            )
        self.val_loss.append(metrics["loss"])
        return metrics["loss"]

    def _batch_loss(self, batch):
        y_pred, label, tracker, _ = self._get_ypred_and_label(batch)
        loss = self.loss(y_pred, label)

        metrics = tracker
        return loss, metrics

    def _extract_batch(self, batch):
        batch = batch.to(self.device)
        fourmomenta = batch.x.to(self.momentum_dtype)
        scalars = batch.scalars.to(self.dtype)
        ptr = batch.ptr
        label = batch.label.to(self.dtype)
        return fourmomenta, scalars, ptr, label

    def _get_ypred_and_label(self, batch):
        fourmomenta, scalars, ptr, label = self._extract_batch(batch)
        embedding = embed_tagging_data(
            fourmomenta,
            scalars,
            ptr,
            self.cfg.data,
        )
        y_pred, tracker, frames = self.model(embedding)
        if isinstance(self.loss, torch.nn.BCEWithLogitsLoss):
            y_pred = y_pred[:, 0]
        return y_pred, label, tracker, frames

    def _init_metrics(self):
        return {
            "reg_collinear": [],
            "reg_coplanar": [],
            "reg_lightlike": [],
            "reg_gammamax": [],
            "gamma_mean": [],
            "gamma_max": [],
        }


class TopTaggingExperiment(TaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = 1
        self.extra_scalars = 0

    def init_data(self):
        data_path = os.path.join(
            self.cfg.data.data_dir, f"toptagging_{self.cfg.data.dataset}.npz"
        )
        self._init_data(TopTaggingDataset, data_path)
