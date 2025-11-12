import hydra
import pytest
import torch
from lloca.utils.rand_transforms import rand_lorentz, rand_rotation

import experiments.logger
from experiments.amplitudes.experiment import AmplitudeExperiment


@pytest.mark.parametrize(
    "model_list",
    list(
        [
            ["model=amp_mlp"],
            ["model=amp_transformer"],
            ["model=amp_graphnet"],
            ["model=amp_graphnet", "model.include_edges=false"],
            ["model=amp_graphnet", "model.include_nodes=false"],
        ],
    ),
)
@pytest.mark.parametrize("framesnet", ["learnedpd", "learnedso13"])
@pytest.mark.parametrize("rand_trafo", [rand_rotation, rand_lorentz])
def test_amplitudes(
    model_list,
    framesnet,
    rand_trafo,
    iterations=1,
):
    experiments.logger.LOGGER.disabled = True  # turn off logging

    # create experiment environment
    with hydra.initialize(config_path="../../config_quick", version_base=None):
        overrides = [
            *model_list,
            f"model/framesnet={framesnet}",
            "save=false",
        ]
        cfg = hydra.compose(config_name="amplitudes", overrides=overrides)
        exp = AmplitudeExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()
    exp.model.eval()
    exp.model.to("cpu")

    def cycle(iterable):
        while True:
            yield from iterable

    mses = []
    iterator = iter(cycle(exp.train_loader))
    for _ in range(iterations):
        data = next(iterator)
        mom = data[1]

        # original data
        amp_original, tracker, _ = exp.model(mom)

        # augmented data
        trafo = rand_trafo(mom.shape[:-2] + (1,), dtype=mom.dtype)
        mom_augmented = torch.einsum("...ij,...j->...i", trafo, mom)
        amp_augmented = exp.model(mom_augmented)[0]

        mse = (amp_original - amp_augmented) ** 2
        mses.append(mse.detach())
    mses = torch.cat(mses, dim=0).flatten().clamp(min=1e-20)
    print(
        f"log-mean={mses.log().mean().exp():.2e} max={mses.max().item():.2e}",
        model_list,
        rand_trafo.__name__,
        framesnet,
    )
    for key in tracker.keys():
        print(f"data tracker key={key}, value={tracker[key]}")
