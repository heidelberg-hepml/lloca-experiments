import pytest
import hydra
from torch.utils.flop_counter import FlopCounterMode

import experiments.logger
from experiments.tagging.experiment import TopTaggingExperiment


@pytest.mark.parametrize(
    "framesnet,equivectors",
    [
        ["identity", None],
        ["learnedpd", "equimlp"],
        ["learnedpd", "pelican"],
        ["learnedpd", "lgatr"],
        ["learnedpd", "lgatr2"],
    ],
)
@pytest.mark.parametrize(
    "model_list",
    [
        ["model=tag_ParT"],
        ["model=tag_particlenet"],
        ["model=tag_transformer"],
        ["model=tag_graphnet"],
        ["model=tag_graphnet", "model.include_edges=true"],
        ["model=tag_lgatr"],
        ["model=tag_MIParT"],
        ["model=tag_MIParT-L"],
        ["model=tag_lorentznet"],
        ["model=tag_pelican_fair"],
    ],
)
def test_tagging(framesnet, model_list, equivectors, jet_size=50):
    experiments.logger.LOGGER.disabled = True  # turn off logging

    # create experiment environment
    with hydra.initialize(config_path="../../config", version_base=None):
        overrides = [
            *model_list,
            f"model/framesnet={framesnet}",
            "save=false",
            "training.batchsize=1",
            "data.dataset=mini",
        ]
        if framesnet != "identity":
            overrides.append(f"model/framesnet/equivectors={equivectors}")
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)
        exp = TopTaggingExperiment(cfg)
    exp._init()
    exp.init_physics()
    try:
        exp.init_model()
    except Exception:
        return
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()

    iterator = iter(exp.train_loader)
    data = next(iterator)
    while data.x.shape[0] < jet_size:
        data = next(iterator)
    data.x = data.x[:jet_size]
    data.scalars = data.scalars[:jet_size]
    data.batch = data.batch[:jet_size]
    data.ptr[-1] = jet_size

    with FlopCounterMode(display=False) as flop_counter:
        exp._get_ypred_and_label(data)
    flops = flop_counter.get_total_flops()
    num_parameters = sum(p.numel() for p in exp.model.parameters())

    print(
        f"flops(batchsize=1)={flops:.2e}; parameters={num_parameters}",
        model_list,
        framesnet,
        equivectors,
    )
    # print(flop_counter.get_table(depth=5))
