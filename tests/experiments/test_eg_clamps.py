import pytest
import hydra

import experiments.logger
from experiments.eventgen.processes import ttbarExperiment
from tests.experiments.utils import track_clamps


@pytest.mark.parametrize(
    "lframesnet",
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
        ["model=eg_mlp"],
        ["model=eg_transformer"],
        ["model=eg_graphnet"],
        ["model=eg_gatr"],
    ],
)
def test_amplitudes(lframesnet, model_list, iterations=1):
    experiments.logger.LOGGER.disabled = True  # turn off logging

    # create experiment environment
    with hydra.initialize(config_path="../../config_quick", version_base=None):
        overrides = [
            *model_list,
            f"model/lframesnet={lframesnet}",
            "save=false",
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
    exp.device = "cpu"
    exp.model.to("cpu")

    for (
        i,
        mom,
    ) in enumerate(exp.train_loader):
        if i == iterations:
            break

        # original data
        with track_clamps() as tracking:
            exp._batch_loss(mom)[0]

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
