import pytest
import hydra

import experiments.logger
from experiments.amplitudes.experiment import AmplitudeExperiment
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
        ["model=amp_mlp"],
        ["model=amp_transformer"],
        ["model=amp_graphnet"],
        ["model=amp_graphnet", "model.include_edges=false"],
        ["model=amp_graphnet", "model.include_nodes=false"],
        ["model=amp_gatr"],
        ["model=amp_dsi"],
    ],
)
@pytest.mark.parametrize("iterations", [1])
def test_amplitudes(framesnet, model_list, iterations):
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
    try:
        exp.init_model()
    except Exception:
        return
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()

    for i, data in enumerate(exp.train_loader):
        mom = data[1]
        if i == iterations:
            break

        # original data
        with track_clamps() as tracking:
            exp.model(mom)[0]

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
