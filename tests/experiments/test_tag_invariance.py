import torch
import pytest
import hydra

import experiments.logger
from experiments.tagging.experiment import TopTaggingExperiment
from lloca.utils.rand_transforms import (
    rand_rotation,
    rand_lorentz,
    rand_xyrotation,
)


BREAKING = [
    "data.beam_reference=null",
    "data.add_time_reference=false",
    "data.tagging_features_framesnet=null",
]


@pytest.mark.parametrize(
    "rand_trafo,breaking_list",
    [
        [rand_rotation, BREAKING],
        [rand_lorentz, BREAKING],
        [rand_xyrotation, []],
    ],
)
@pytest.mark.parametrize(
    "model_list",
    list(
        [
            ["model=tag_ParT"],
            ["model=tag_transformer"],
            ["model=tag_graphnet"],
            ["model=tag_graphnet", "model.include_edges=true"],
        ]
    ),
)
@pytest.mark.parametrize("framesnet", ["learnedpd", "learnedso13"])
def test_amplitudes(
    rand_trafo,
    model_list,
    framesnet,
    breaking_list,
    iterations=1,
    batchsize=4,
):
    experiments.logger.LOGGER.disabled = True  # turn off logging

    # create experiment environment
    with hydra.initialize(config_path="../../config_quick", version_base=None):
        overrides = [
            *model_list,
            f"model/framesnet={framesnet}",
            f"training.batchsize={batchsize}",
            "save=false",
            *breaking_list,
        ]
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)
        exp = TopTaggingExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()
    exp.model.eval()  # turn off dropout

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    mses = []
    iterator = iter(cycle(exp.train_loader))
    for _ in range(iterations):
        data = next(iterator)
        data_augmented = data.clone()

        # original data
        y_pred, _, tracker, _ = exp._get_ypred_and_label(data)

        # augmented data
        mom = data_augmented.x
        trafo = rand_trafo(mom.shape[:-2] + (1,), dtype=mom.dtype)
        mom_augmented = torch.einsum("...ij,...j->...i", trafo, mom)
        data_augmented.x = mom_augmented
        y_pred_augmented = exp._get_ypred_and_label(data_augmented)[0]

        mse = (y_pred_augmented - y_pred) ** 2
        mses.append(mse.detach())
    mses = torch.cat(mses, dim=0).clamp(min=1e-20)
    print(
        f"log-mean={mses.log().mean().exp():.2e} max={mses.max().item():.2e}",
        model_list,
        rand_trafo.__name__,
        framesnet,
    )
    for key in tracker.keys():
        print(f"data tracker key={key}, value={tracker[key]}")
