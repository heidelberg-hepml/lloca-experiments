import pytest
import hydra
import torch
from torch.utils.flop_counter import FlopCounterMode

import experiments.logger
from experiments.tagging.experiment import TopTaggingExperiment


@pytest.mark.parametrize("lframesnet", ["identity", "polardec"])
@pytest.mark.parametrize(
    "model_list",
    [
        ["model=tag_ParT"],
        ["model=tag_particlenet"],
        ["model=tag_particlenet-lite"],
        ["model=tag_transformer"],
        ["model=tag_graphnet"],
        ["model=tag_graphnet", "model.include_edges=true"],
        ["model=tag_gatr"],
        ["model=tag_MIParT"],
        ["model=tag_MIParT-L"],
    ],
)
def test_tagging(lframesnet, model_list, jet_size=50):
    experiments.logger.LOGGER.disabled = True  # turn off logging

    # create experiment environment
    with hydra.initialize(config_path="../../config", version_base=None):
        overrides = [
            *model_list,
            f"model/lframesnet={lframesnet}",
            "save=false",
            "training.batchsize=1",
            "data.dataset=mini",
        ]
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)
        exp = TopTaggingExperiment(cfg)
    exp._init()
    exp.init_physics()
    try:
        exp.init_model()
    except Exception as e:
        return
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()

    data = next(iter(exp.train_loader))

    # fill batch with dummy data of fixed length
    data.x = torch.ones(jet_size, 4, dtype=data.x.dtype)
    data.scalars = torch.ones(jet_size, data.scalars.shape[1], dtype=data.scalars.dtype)
    data.batch = torch.zeros(jet_size, dtype=data.batch.dtype)
    data.ptr[-1] = jet_size

    with FlopCounterMode(display=False) as flop_counter:
        exp._get_ypred_and_label(data)
    flops = flop_counter.get_total_flops()
    num_parameters = sum(p.numel() for p in exp.model.parameters())

    print(
        f"flops(batchsize=1)={flops:.2e}; parameters={num_parameters}",
        model_list,
        lframesnet,
    )
    # print(flop_counter.get_table(depth=2))
