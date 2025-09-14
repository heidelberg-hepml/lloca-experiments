import torch
import pytest
import hydra

import experiments.logger
from experiments.eventgen.processes import ttbarExperiment
from lloca.utils.transforms import rand_rotation, rand_lorentz, rand_xyrotation

BREAKING = [
    "data.spurions.beam_reference=null",
    "data.spurions.add_time_reference=false",
    "data.spurions.scalar_dims=[]",
]


@pytest.mark.parametrize(
    "model_list",
    list(
        [
            ["model=eg_mlp"],
            ["model=eg_transformer"],
            ["model=eg_graphnet"],
        ],
    ),
)
@pytest.mark.parametrize("framesnet", ["polardec", "orthogonal"])
@pytest.mark.parametrize(
    "rand_trafo,breaking_list",
    [
        [rand_rotation, BREAKING],
        [rand_lorentz, BREAKING],
        [rand_xyrotation, []],
    ],
)
def test_amplitudes(
    model_list,
    framesnet,
    rand_trafo,
    breaking_list,
    iterations=1,
    batchsize=64,
):
    experiments.logger.LOGGER.disabled = True  # turn off logging

    # create experiment environment
    with hydra.initialize(config_path="../../config_quick", version_base=None):
        overrides = [
            *model_list,
            f"model/framesnet={framesnet}",
            f"training.batchsize={batchsize}",
            "save=false",
            "cfm.coordinates=Fourmomenta",
            *breaking_list,
        ]
        cfg = hydra.compose(config_name="ttbar", overrides=overrides)
        exp = ttbarExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()
    exp.model.eval()
    exp.model.to("cpu")
    exp.device = "cpu"

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    mses = []
    iterator = iter(cycle(exp.train_loader))
    for _ in range(iterations):
        data = next(iterator)
        mom_original = data[0]

        # mimic batch_loss in cfm.py
        fm0 = mom_original
        t = torch.rand(fm0.shape[0], 1, 1)
        fm1 = exp.model.sample_base(
            fm0.shape, device=torch.device("cpu"), dtype=torch.float32
        )
        x0 = exp.model.coordinates.fourmomenta_to_x(fm0)
        x1 = exp.model.coordinates.fourmomenta_to_x(fm1)
        xt = exp.model.geometry.get_trajectory(x0, x1, t)[0]

        # sample data augmentation
        trafo = rand_trafo(mom_original.shape[:-2] + (1,), dtype=mom_original.dtype)

        # model + augmentation
        v_original = exp.model.get_velocity(xt, t)[0]
        v_original = torch.einsum("...ij,...j->...i", trafo, v_original)

        # augmentation + model
        xt_augmented = torch.einsum("...ij,...j->...i", trafo, xt)
        v_augmented = exp.model.get_velocity(xt_augmented, t)[0]

        mse = (v_original - v_augmented) ** 2
        mses.append(mse.detach())
    mses = torch.cat(mses, dim=0).flatten().clamp(min=1e-20)
    print(
        f"log-mean={mses.log().mean().exp():.2e} max={mses.max().item():.2e}",
        model_list,
        rand_trafo.__name__,
        framesnet,
    )
