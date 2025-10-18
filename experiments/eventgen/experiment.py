import numpy as np
import torch

import os, time
from omegaconf import open_dict
from hydra.utils import instantiate
from tqdm import trange, tqdm

from experiments.base_experiment import BaseExperiment
from experiments.eventgen.utils import ensure_onshell
import experiments.eventgen.plotter as plotter
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow


class EventGenerationExperiment(BaseExperiment):
    def init_physics(self):
        self.define_process_specifics()

        # dynamically set wrapper properties
        self.modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
        n_particles = self.n_hard_particles + self.cfg.data.n_jets
        with open_dict(self.cfg):
            self.cfg.model.n_particles = n_particles
            if self.modelname in ["Transformer", "GraphNet"]:
                self.cfg.model.net.in_channels = (
                    n_particles + self.cfg.cfm.embed_t_dim + 4
                )
                self.cfg.model.net.out_channels = 4 + len(
                    self.cfg.data.spurions.scalar_dims
                )
                if self.modelname == "GraphNet":
                    self.cfg.model.net.num_edge_attr = (
                        1 if self.cfg.model.include_edges else 0
                    )

            elif self.modelname == "MLP":
                self.cfg.model.net.in_shape = self.cfg.cfm.embed_t_dim + 4 * n_particles
                self.cfg.model.net.out_shape = n_particles * (
                    4 + len(self.cfg.data.spurions.scalar_dims)
                )

            elif self.modelname == "LGATr":
                self.cfg.model.net.in_s_channels = (
                    n_particles + self.cfg.cfm.embed_t_dim
                )
                self.cfg.model.net.in_mv_channels = (
                    2 if self.cfg.data.spurions.add_time_reference else 1
                )
                if self.cfg.data.spurions.beam_reference is not None:
                    self.cfg.model.net.in_mv_channels += (
                        2 if self.cfg.data.spurions.two_beams else 1
                    )
                self.cfg.model.net.out_s_channels = len(
                    self.cfg.data.spurions.scalar_dims
                )

            else:
                raise NotImplementedError

            # copy model-specific parameters
            self.cfg.model.network_float64 = self.cfg.use_float64
            self.cfg.model.momentum_float64 = self.cfg.data.momentum_float64
            self.cfg.model.odeint = self.cfg.odeint
            self.cfg.model.cfm = self.cfg.cfm
            self.cfg.model.spurions = self.cfg.data.spurions

        # decide which entries to use for the framesnet
        if "equivectors" in self.cfg.model.framesnet:
            self.cfg.model.framesnet.equivectors.num_scalars = (
                n_particles + self.cfg.cfm.embed_t_dim
            )

    def init_data(self):
        LOGGER.info(f"Working with {self.cfg.data.n_jets} extra jets")
        momentum_dtype = (
            torch.float64 if self.cfg.data.momentum_float64 else torch.float32
        )

        # load data
        data_path = eval(f"self.cfg.data.data_path_{self.cfg.data.n_jets}j")
        assert os.path.exists(data_path), f"data_path {data_path} does not exist"
        data_raw = np.load(data_path)
        LOGGER.info(f"Loaded data with shape {data_raw.shape} from {data_path}")

        # bring data into correct shape
        if self.cfg.data.subsample is not None:
            assert self.cfg.data.subsample < data_raw.shape[0]
            LOGGER.info(
                f"Reducing the size of the dataset from {data_raw.shape[0]} to {self.cfg.data.subsample}"
            )
            data_raw = data_raw[: self.cfg.data.subsample, :]
        data_raw = data_raw.reshape(data_raw.shape[0], data_raw.shape[1] // 4, 4)
        data_raw = torch.tensor(data_raw, dtype=momentum_dtype)

        # collect everything
        self.events_raw = data_raw

        # change global units
        self.model.init_physics(
            self.units,
            self.pt_min,
            self.delta_r_min,
            self.onshell_list,
            self.onshell_mass,
            self.virtual_components,
            self.cfg.data.base_type,
            self.cfg.data.use_pt_min,
            self.cfg.data.use_delta_r_min,
        )

        # preprocessing
        self.events_raw = ensure_onshell(
            self.events_raw,
            self.onshell_list,
            self.onshell_mass,
        )
        self.events_prepd = self.model.preprocess(self.events_raw)

        # initialize cfm (might require data)
        self.model.init_distribution()
        self.model.init_coordinates()
        self.model.coordinates.init_fit(self.events_prepd)
        self.model.distribution.coordinates.init_fit(self.events_prepd)
        self.model.init_geometry()
        if hasattr(self.model, "init_standardization"):
            self.model.init_standardization(self.events_prepd)

    def _init_dataloader(self):
        assert sum(self.cfg.data.train_test_val) <= 1

        # seperate data into train, test and validation subsets
        n_data = self.events_raw.shape[0]
        split_val = int(n_data * self.cfg.data.train_test_val[::-1][0])
        split_test = int(n_data * sum(self.cfg.data.train_test_val[::-1][:2]))
        split_train = int(n_data * sum(self.cfg.data.train_test_val[::-1]))

        self.data_raw = {
            "val": self.events_raw[0:split_val],
            "tst": self.events_raw[split_val:split_test],
            "trn": self.events_raw[split_test:split_train],
        }
        self.data_prepd = {
            "val": self.events_prepd[0:split_val],
            "tst": self.events_prepd[split_val:split_test],
            "trn": self.events_prepd[split_test:split_train],
        }

        # create dataloaders
        trn_dataset = torch.utils.data.TensorDataset(self.data_prepd["trn"])
        tst_dataset = torch.utils.data.TensorDataset(self.data_prepd["tst"])
        val_dataset = torch.utils.data.TensorDataset(self.data_prepd["val"])

        trn_sampler = torch.utils.data.DistributedSampler(
            trn_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        val_sampler = torch.utils.data.DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )
        tst_sampler = torch.utils.data.DistributedSampler(
            tst_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
        )

        self.train_loader = torch.utils.data.DataLoader(
            dataset=trn_dataset,
            batch_size=self.cfg.training.batchsize // self.world_size,
            sampler=trn_sampler,
            drop_last=len(trn_dataset) >= self.cfg.training.batchsize,
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=tst_dataset,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            sampler=tst_sampler,
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=self.cfg.evaluation.batchsize // self.world_size,
            sampler=val_sampler,
        )

        LOGGER.info(
            f"Constructed dataloaders with train_test_val={self.cfg.data.train_test_val}, "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

    def _init_optimizer(self, param_groups=None):
        # handle linear layer in the t_embedding in the same way as the main net
        # this could be handled in more elegant way
        param_groups = [
            {
                "params": self.model.net.parameters(),
                "lr": self.cfg.training.lr,
                "weight_decay": self.cfg.training.weight_decay,
            },
            {
                "params": self.model.framesnet.parameters(),
                "lr": self.cfg.training.lr_factor_framesnet * self.cfg.training.lr,
                "weight_decay": self.cfg.training.weight_decay_framesnet,
            },
            {
                "params": self.model.t_embedding.parameters(),
                "lr": self.cfg.training.lr,
                "weight_decay": self.cfg.training.weight_decay,
            },
        ]
        super()._init_optimizer(param_groups=param_groups)

    @torch.no_grad()
    def evaluate(self):
        if self.ema is not None:
            # no EMA + no-EMA evaluation implemented for generation
            LOGGER.info(f"Evaluating with ema")
            with self.ema.average_parameters():
                self.evaluate_inner()
        else:
            self.evaluate_inner()

    def evaluate_inner(self):
        loaders = {
            "trn": self.train_loader,
            "tst": self.test_loader,
            "val": self.val_loader,
        }
        if self.cfg.evaluation.sample:
            self._sample_events()
            loaders["gen"] = self.sample_loader
        else:
            LOGGER.info("Skip sampling")

        if self.cfg.evaluation.classifier:
            self.classifiers = self._evaluate_classifier_metric()

        for key in self.cfg.evaluation.eval_log_prob:
            if key == "gen":
                # log_probs of generated events are not interesting
                # + they are not well-defined, because generated events might be in regions
                # that are not included in the base distribution (because of pt_min, delta_r_min)
                continue
            self._evaluate_log_prob_single(loaders[key], key)

    def _evaluate_classifier_metric(self):
        assert self.cfg.evaluation.sample, "need samples for classifier evaluation"

        # initiate
        with open_dict(self.cfg):
            num_particles = self.n_hard_particles + self.cfg.data.n_jets
            self.cfg.classifier.net.in_shape = 4 * num_particles
            if self.cfg.classifier.cfg_preprocessing.add_delta_r:
                self.cfg.classifier.net.in_shape += (
                    num_particles * (num_particles - 1) // 2
                )
            if self.cfg.classifier.cfg_preprocessing.add_virtual:
                self.cfg.classifier.net.in_shape += 4 * len(self.virtual_components)
        classifier_factory = instantiate(self.cfg.classifier, _partial_=True)
        classifier = classifier_factory(experiment=self, device=self.device)

        data_true = self.events_raw
        data_fake = self.data_raw["gen"]
        LOGGER.info(
            f"Classifier generated data true/fake has shape {tuple(data_true.shape)}/{tuple(data_fake.shape)}"
        )

        # preprocessing
        cls_params = {"mean": None, "std": None}
        data_true, cls_params = classifier.preprocess(data_true, cls_params)
        data_fake = classifier.preprocess(data_fake, cls_params)[0]
        data_true, data_fake = data_true.to(self.dtype), data_fake.to(self.dtype)
        data_true = classifier.train_test_val_split(data_true)
        data_fake = classifier.train_test_val_split(data_fake)
        classifier.init_data(data_true, data_fake)

        # do things
        classifier.train()
        classifier.evaluate()

        # save weighted events
        if self.cfg.evaluation.save_samples and self.cfg.save:
            weights_true = classifier.results["weights"]["true"]
            weights_fake = classifier.results["weights"]["fake"]

            n_data = self.events_raw.shape[0]
            split_val = int(n_data * self.cfg.data.train_test_val[::-1][0])
            split_test = int(n_data * sum(self.cfg.data.train_test_val[::-1][:2]))
            split_train = int(n_data * sum(self.cfg.data.train_test_val[::-1]))
            os.makedirs(os.path.join(self.cfg.run_dir, "samples"), exist_ok=True)

            filename = os.path.join(
                self.cfg.run_dir,
                "samples",
                f"classifier_{self.cfg.run_idx}",
            )
            np.savez(
                filename,
                events_test=self.events_raw[split_val:split_test],
                events_train=self.events_raw[split_test:split_train],
                events_fake=self.data_raw["gen"],
                weights_test=weights_true[split_val:split_test],
                weights_train=weights_true[split_test:split_train],
                weights_fake=weights_fake,
            )
        return classifier

    def _evaluate_log_prob_single(self, loader, title):
        self.model.eval()
        self.NLLs = []
        LOGGER.info(f"Starting to evaluate log_prob for model on {title} dataset")
        t0 = time.time()
        for i, data in enumerate(tqdm(loader)):
            data = data[0].to(self.device)
            NLL = -self.model.log_prob(data).squeeze().cpu()
            self.NLLs.extend(NLL.squeeze().numpy().tolist())
        dt = time.time() - t0
        LOGGER.info(
            f"Finished evaluating log_prob for {title} dataset after {dt/60:.2f}min"
        )
        LOGGER.info(f"NLL = {np.mean(self.NLLs)}")
        if self.cfg.use_mlflow:
            log_mlflow(f"eval.{title}.NLL", np.mean(self.NLLs))

    def _sample_events(self):
        self.model.eval()

        sample = []
        shape = (
            self.cfg.evaluation.batchsize,
            self.n_hard_particles + self.cfg.data.n_jets,
            4,
        )
        n_batches = (
            1 + (self.cfg.evaluation.nsamples - 1) // self.cfg.evaluation.batchsize
        )
        LOGGER.info(f"Starting to generate {self.cfg.evaluation.nsamples} events")
        t0 = time.time()
        for i in trange(n_batches, desc="Sampled batches"):
            x_t = self.model.sample(shape, self.device)
            sample.append(x_t)
        t1 = time.time()
        LOGGER.info(
            f"Finished generating events after {t1-t0:.2f}s = {(t1-t0)/60:.2f}min"
        )

        samples = torch.cat(sample, dim=0)[: self.cfg.evaluation.nsamples, ...].cpu()
        self.data_prepd["gen"] = samples

        samples_raw = self.model.undo_preprocess(samples)
        self.data_raw["gen"] = samples_raw

        if (
            self.cfg.evaluation.save_samples
            and self.cfg.save
            and not self.cfg.evaluation.classifier
        ):
            os.makedirs(os.path.join(self.cfg.run_dir, "samples"), exist_ok=True)
            filename = os.path.join(
                self.cfg.run_dir,
                "samples",
                f"samples_{self.cfg.run_idx}.npy",
            )
            np.save(filename, samples_raw)

        self.sample_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(self.data_prepd["gen"]),
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

    def plot(self):
        path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(os.path.join(path), exist_ok=True)
        LOGGER.info(f"Creating plots in {path}")

        kwargs = {
            "exp": self,
            "model_label": self.modelname,
        }

        if self.cfg.train:
            filename = os.path.join(path, "training.pdf")
            plotter.plot_losses(filename=filename, **kwargs)

        if not self.cfg.evaluate:
            return

        # set correct masses
        if self.cfg.evaluation.sample:
            for label in ["trn", "tst", "gen"]:
                self.data_raw[label] = ensure_onshell(
                    self.data_raw[label],
                    self.onshell_list,
                    self.onshell_mass,
                )

        # If specified, collect weights from classifier
        if self.cfg.evaluation.classifier and self.cfg.plotting.reweighted:
            weights = self.classifiers.weights_fake.squeeze()
        else:
            weights = None

        # can manually create a mask
        if self.cfg.plotting.create_mask:
            assert weights is not None
            mask_dict = {"condition": "w<0.5", "mask": weights < 0.5}
            weights = None
        else:
            mask_dict = None

        if (
            self.cfg.plotting.log_prob
            and len(self.cfg.evaluation.eval_log_prob) > 0
            and self.cfg.evaluate
        ):
            filename = os.path.join(path, "neg_log_prob.pdf")
            plotter.plot_log_prob(filename=filename, **kwargs)

        if self.cfg.evaluation.classifier and self.cfg.evaluate:
            filename = os.path.join(path, "classifier.pdf")
            plotter.plot_classifier(filename=filename, **kwargs)

        if self.cfg.evaluation.sample:
            if self.cfg.plotting.fourmomenta:
                filename = os.path.join(path, "fourmomenta.pdf")
                plotter.plot_fourmomenta(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )

            if self.cfg.plotting.jetmomenta:
                filename = os.path.join(path, "jetmomenta.pdf")
                plotter.plot_jetmomenta(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )

            if self.cfg.plotting.preprocessed:
                filename = os.path.join(path, "preprocessed.pdf")
                plotter.plot_preprocessed(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )

            if self.cfg.plotting.virtual and len(self.virtual_components) > 0:
                filename = os.path.join(path, "virtual.pdf")
                plotter.plot_virtual(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )

            if self.cfg.plotting.delta:
                filename = os.path.join(path, "delta.pdf")
                plotter.plot_delta(
                    filename=filename, **kwargs, weights=weights, mask_dict=mask_dict
                )

            if self.cfg.plotting.deta_dphi:
                filename = os.path.join(path, "deta_dphi.pdf")
                plotter.plot_deta_dphi(filename=filename, **kwargs)

    def _init_loss(self):
        # loss defined manually within the model
        pass

    def _batch_loss(self, data):
        data = data[0].to(self.device)
        loss, metrics = self.model.batch_loss(data)
        return loss, metrics

    def _init_metrics(self):
        metrics = {
            "reg_collinear": [],
            "reg_coplanar": [],
            "reg_lightlike": [],
            "reg_gammamax": [],
            "gamma_mean": [],
            "gamma_max": [],
        }
        for k in range(4):
            metrics[f"mse_{k}"] = []
        return metrics

    def define_process_specifics(self):
        self.plot_title = None
        self.n_hard_particles = None
        self.n_jets_max = None
        self.onshell_list = None
        self.onshell_mass = None
        self.units = None
        self.pt_min = None
        self.delta_r_min = None
        self.obs_names_index = None
        self.fourmomentum_ranges = None
        self.jetmomentum_ranges = None
        self.virtual_components = None
        self.virtual_names = None
        self.virtual_ranges = None
        raise NotImplementedError
