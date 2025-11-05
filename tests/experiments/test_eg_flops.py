import pytest
import hydra
from torch.utils.flop_counter import FlopCounterMode

import experiments.logger
from experiments.eventgen.processes import ttbarExperiment


@pytest.mark.parametrize(
    "framesnet",
    [
        "identity",
        "learnedpd",
    ],
)
@pytest.mark.parametrize(
    "model_list",
    [
        ["model=eg_mlp"],
        ["model=eg_transformer"],
        ["model=eg_graphnet"],
        ["model=eg_lgatr"],
    ],
)
def test_amplitudes(framesnet, model_list, iterations=1):
    experiments.logger.LOGGER.disabled = True  # turn off logging

    # create experiment environment
    with hydra.initialize(config_path="../../config", version_base=None):
        overrides = [
            *model_list,
            f"model/framesnet={framesnet}",
            "save=false",
            "training.batchsize=1",
            "data.data_path_0j=data/ttbar_0j_mini.npy",
        ]
        cfg = hydra.compose(config_name="ttbar", overrides=overrides)
        exp = ttbarExperiment(cfg)
    exp._init()
    exp.init_physics()
    try:
        exp.init_model()
    except Exception:
        return
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()

    data = next(iter(exp.train_loader))
    with FlopCounterMode(display=False) as flop_counter:
        exp._batch_loss(data)
    flops = flop_counter.get_total_flops()
    num_parameters = sum(p.numel() for p in exp.model.parameters())

    print(
        f"flops(batchsize=1)={flops:.2e}; parameters={num_parameters}",
        model_list,
        framesnet,
    )
    # print(flop_counter.get_table(depth=2))
