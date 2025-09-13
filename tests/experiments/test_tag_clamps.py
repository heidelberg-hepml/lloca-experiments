import pytest
import hydra

import experiments.logger
from experiments.tagging.experiment import TopTaggingExperiment
from tests.experiments.utils import track_clamps


@pytest.mark.parametrize(
    "framesnet",
    [
        "identity",
        "randomrotation",
        "randomlorentz",
        "orthogonal",
        "polardec",
    ],
)
@pytest.mark.parametrize(
    "model_list",
    [
        ["model=tag_ParT"],
        ["model=tag_particlenet-lite"],
        ["model=tag_transformer"],
        ["model=tag_graphnet"],
        ["model=tag_graphnet", "model.include_edges=true"],
        ["model=tag_gatr"],
    ],
)
def test_tagging(framesnet, model_list, iterations=1):
    # Note: ParticleTransformer clamps log(kT) very often,
    # this happens already in non-equivariant models
    experiments.logger.LOGGER.disabled = True  # turn off logging

    # create experiment environment
    with hydra.initialize(config_path="../../config_quick", version_base=None):
        overrides = [
            *model_list,
            f"model/framesnet={framesnet}",
            "save=false",
            "data.beam_reference=null",
            "data.add_time_reference=false",
        ]
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

    for i, data in enumerate(exp.train_loader):
        if i == iterations:
            break

        # original data
        with track_clamps() as tracking:
            exp._get_ypred_and_label(data)[0]

        sorted_calls = sorted(
            tracking,
            key=lambda c: c["num_elements_clamped"] / c["total_elements"],
            reverse=True,
        )

        for i, c in enumerate(sorted_calls):
            if c["num_elements_clamped"] == 0:
                continue

            ratio = c["num_elements_clamped"] / c["total_elements"]
            print(
                f"{i+1}. {c['op_type']} clamped {c['num_elements_clamped']} / {c['total_elements']} "
                f"({ratio:.1%}) at {c['filename']}:{c['line']}"
            )
            print(f"   → {c['code']}")
            assert c["num_elements_clamped"] == 0
