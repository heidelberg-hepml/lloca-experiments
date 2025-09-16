import hydra
from experiments.tagging.experiment import TopTaggingExperiment
from experiments.tagging.finetuneexperiment import TopTaggingFineTuneExperiment
from experiments.tagging.topxlexperiment import TopXLTaggingExperiment
from experiments.tagging.jetclassexperiment import JetClassTaggingExperiment
from experiments.amplitudes.experiment import AmplitudeExperiment
from experiments.amplitudes.experimentxl import AmplitudeXLExperiment
from experiments.eventgen.processes import ttbarExperiment


@hydra.main(config_path="config_quick", config_name="toptagging", version_base=None)
def main(cfg):
    if cfg.exp_type == "toptagging":
        exp = TopTaggingExperiment(cfg)
    elif cfg.exp_type == "toptaggingft":
        exp = TopTaggingFineTuneExperiment(cfg)
    elif cfg.exp_type == "topxltagging":
        exp = TopXLTaggingExperiment(cfg)
    elif cfg.exp_type == "jctagging":
        exp = JetClassTaggingExperiment(cfg)
    elif cfg.exp_type == "amplitudes":
        exp = AmplitudeExperiment(cfg)
    elif cfg.exp_type == "amplitudesxl":
        exp = AmplitudeXLExperiment(cfg)
    elif cfg.exp_type == "ttbar":
        exp = ttbarExperiment(cfg)
    else:
        raise ValueError(f"exp_type {cfg.exp_type} not implemented")

    exp()


if __name__ == "__main__":
    main()
