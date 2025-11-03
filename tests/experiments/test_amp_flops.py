# Should be evaluated on GPU
# otherwise the transformer FLOPs will be off, because it is not using flash-attention
import pytest
import hydra
from torch.utils.flop_counter import FlopCounterMode

import experiments.logger
from experiments.amplitudes.experiment import AmplitudeExperiment


@pytest.mark.parametrize(
    "framesnet,equivectors",
    [
        ["identity", None],
        ["learnedpd", "equimlp"],
        ["learnedpd", "pelican"],
        ["learnedpd", "lgatr"],
    ],
)
@pytest.mark.parametrize(
    "model_list",
    [
        ["model=amp_mlp"],
        ["model=amp_transformer"],
        ["model=amp_graphnet"],
        ["model=amp_graphnet", "model.include_edges=false"],
        ["model=amp_graphnet", "model.include_nodes=false"],
        ["model=amp_lgatr"],
        ["model=amp_dsi"],
    ],
)
def test_amplitudes(framesnet, model_list, equivectors):
    experiments.logger.LOGGER.disabled = True  # turn off logging

    # create experiment environment
    with hydra.initialize(config_path="../../config", version_base=None):
        overrides = [
            *model_list,
            f"model/framesnet={framesnet}",
            "save=false",
            "training.batchsize=1",
            "data.dataset=zgggg_mini",
        ]
        if framesnet != "identity":
            overrides.append(f"model/framesnet/equivectors={equivectors}")
        cfg = hydra.compose(config_name="amplitudes", overrides=overrides)
        exp = AmplitudeExperiment(cfg)
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
    mom = data[1].to(device=exp.device, dtype=exp.dtype)

    with FlopCounterMode(display=False) as flop_counter:
        exp.model(mom)
    flops = flop_counter.get_total_flops()
    num_parameters = sum(p.numel() for p in exp.model.parameters())

    print(
        f"flops(batchsize=1)={flops:.2e}; parameters={num_parameters}",
        model_list,
        framesnet,
        equivectors,
    )
    # print(flop_counter.get_table(depth=2))
