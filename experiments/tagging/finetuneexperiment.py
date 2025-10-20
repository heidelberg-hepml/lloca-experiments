import os, torch
from omegaconf import OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from torch_ema import ExponentialMovingAverage

from experiments.tagging.experiment import TopTaggingExperiment
from experiments.logger import LOGGER


def _extract_cli_overrides(cfg, prefix):
    if not HydraConfig.initialized():
        return OmegaConf.create()
    out = OmegaConf.create()
    for s in HydraConfig.get().overrides.task:
        s_ = s.lstrip("+~")  # handle +add / ~delete syntax
        if not s_.startswith(prefix):
            continue
        key = s_.split("=", 1)[0]  # absolute path, e.g. model.foo.bar
        val = OmegaConf.select(cfg, key)
        rel = key[len(prefix) :]  # inside subtree
        OmegaConf.update(out, rel, val, merge=True)
    return out


class TopTaggingFineTuneExperiment(TopTaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # load warm_start cfg
        warmstart_path = os.path.join(
            self.cfg.finetune.backbone_path, self.cfg.finetune.backbone_cfg
        )
        self.warmstart_cfg = OmegaConf.load(warmstart_path)
        assert self.warmstart_cfg.exp_type in ["jctagging", "topxltagging"]
        assert self.warmstart_cfg.data.features == "fourmomenta"

        if not self.warmstart_cfg.model._target_ in [
            "experiments.tagging.wrappers.TransformerWrapper",
            "experiments.tagging.wrappers.ParTWrapper",
        ]:
            raise NotImplementedError

        # merge config files
        with open_dict(self.cfg):
            # model: warmstart defaults, overridden only by CLI `model.*`
            model_cli = _extract_cli_overrides(self.cfg, "model.")
            self.cfg.model = OmegaConf.merge(self.warmstart_cfg.model, model_cli)

            self.cfg.ema = self.warmstart_cfg.ema

            # overwrite model-specific cfg.data entries
            # NOTE: might have to extend this if adding more models
            self.cfg.data.add_tagging_features_framesnet = (
                self.warmstart_cfg.data.add_tagging_features_framesnet
            )
            self.cfg.data.beam_reference = self.warmstart_cfg.data.beam_reference
            self.cfg.data.two_beams = self.warmstart_cfg.data.two_beams
            self.cfg.data.add_time_reference = (
                self.warmstart_cfg.data.add_time_reference
            )
            self.cfg.data.mass_reg = self.warmstart_cfg.data.mass_reg
            self.cfg.data.spurion_scale = self.warmstart_cfg.data.spurion_scale
            self.cfg.data.momentum_float64 = self.warmstart_cfg.data.momentum_float64

    def init_model(self):
        # overwrite output channel shape to allow loading pretrained weights
        self.cfg.model.out_channels = self.warmstart_cfg.model.out_channels

        super().init_model()

        if self.warm_start:
            # nothing to do
            return

        # load pretrained weights
        model_path = os.path.join(
            self.warmstart_cfg.run_dir,
            "models",
            f"model_run{self.warmstart_cfg.run_idx}.pt",
        )
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)[
                "model"
            ]
        except FileNotFoundError:
            raise ValueError(f"Cannot load model from {model_path}")
        LOGGER.info(f"Loading pretrained model from {model_path}")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device, dtype=self.dtype)

        # overwrite output layer
        if (
            self.warmstart_cfg.model._target_
            == "experiments.tagging.wrappers.TransformerWrapper"
        ):
            self.model.net.linear_out = torch.nn.Linear(
                self.model.net.hidden_channels, self.num_outputs
            ).to(self.device)
        elif (
            self.warmstart_cfg.model._target_
            == "experiments.tagging.wrappers.ParTWrapper"
        ):
            # overwrite output layer, reset parameters for all other layers in the final MLP
            self.model.net.fc[-1] = torch.nn.Linear(
                self.model.net.embed_dim, self.num_outputs
            ).to(self.device)
            for module in self.model.net.fc.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        else:
            raise NotImplementedError

        if self.cfg.ema:
            LOGGER.info(f"Re-initializing EMA")
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=self.cfg.training.ema_decay
            ).to(self.device)

    def _init_optimizer(self):
        # collect parameter lists
        if (
            self.warmstart_cfg.model._target_
            == "experiments.tagging.wrappers.TransformerWrapper"
        ):
            params_backbone_lfnet = list(self.model.framesnet.parameters())
            params_backbone_main = list(self.model.net.linear_in.parameters()) + list(
                self.model.net.blocks.parameters()
            )
            params_head = self.model.net.linear_out.parameters()

            # assign parameter-specific learning rates
            param_groups = [
                {
                    "params": params_backbone_lfnet,
                    "lr": self.cfg.finetune.lr_backbone
                    * self.cfg.training.lr_factor_framesnet,
                    "weight_decay": self.cfg.training.weight_decay_framesnet,
                },
                {
                    "params": params_backbone_main,
                    "lr": self.cfg.finetune.lr_backbone,
                    "weight_decay": self.cfg.training.weight_decay,
                },
                {
                    "params": params_head,
                    "lr": self.cfg.finetune.lr_head,
                    "weight_decay": self.cfg.training.weight_decay,
                },
            ]
        elif (
            self.warmstart_cfg.model._target_
            == "experiments.tagging.wrappers.ParTWrapper"
        ):
            # adapted version of the basic _init_optimizer() in TaggingExperiment
            decay, no_decay, head_decay, head_nodecay = {}, {}, {}, {}
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
                    if name.startswith("fc."):
                        head_nodecay[name] = param
                    else:
                        no_decay[name] = param
                else:
                    if name.startswith("fc."):
                        head_decay[name] = param
                    else:
                        decay[name] = param
            decay_1x, no_decay_1x = list(decay.values()), list(no_decay.values())
            head_decay_1x, head_nodecay_1x = (
                list(head_decay.values()),
                list(head_nodecay.values()),
            )
            param_groups = [
                {
                    "params": no_decay_1x,
                    "weight_decay": 0.0,
                    "lr": self.cfg.finetune.lr_backbone,
                },
                {
                    "params": decay_1x,
                    "weight_decay": self.cfg.training.weight_decay,
                    "lr": self.cfg.finetune.lr_backbone,
                },
                {
                    "params": self.model.framesnet.parameters(),
                    "weight_decay": self.cfg.training.weight_decay_framesnet,
                    "lr": self.cfg.finetune.lr_backbone
                    * self.cfg.training.lr_factor_framesnet,
                },
                {
                    "params": head_nodecay_1x,
                    "weight_decay": 0.0,
                    "lr": self.cfg.finetune.lr_head,
                },
                {
                    "params": head_decay_1x,
                    "weight_decay": self.cfg.training.weight_decay,
                    "lr": self.cfg.finetune.lr_head,
                },
            ]
        else:
            raise NotImplementedError

        super()._init_optimizer(param_groups=param_groups)
