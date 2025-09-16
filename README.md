<div align="center">

## Lorentz Local Canonicalization: How to Make Any Network Lorentz-Equivariant

[![pytorch](https://img.shields.io/badge/PyTorch_2.2+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)

[![LLoCa-CS](http://img.shields.io/badge/paper-arxiv.2505.20280-B31B1B.svg)](https://arxiv.org/abs/2505.20280)
[![LLoCa-HEP](http://img.shields.io/badge/paper-arxiv.2508.14898-B31B1B.svg)](https://arxiv.org/abs/2508.14898)

</div>

This repository contains the official implementation of [Lorentz Local Canonicalization: How to make any Network Lorentz-Equivariant](https://arxiv.org/abs/2505.20280) by [Jonas Spinner](mailto:j.spinner@thphys.uni-heidelberg.de), [Luigi Favaro](mailto:luigi.favaro@uclouvain.be), [Peter Lippmann](mailto:peter.lippmann@iwr.uni-heidelberg.de), [Sebastian Pitz](mailto:pitz@thphys.uni-heidelberg.de), [Gerrit Gerhartz](mailto:gerhartz@thphys.uni-heidelberg.de), Tilman Plehn, and Fred A. Hamprecht.

## 1. Getting started

Clone the repository.

```bash
git clone https://github.com/heidelberg-hepml/lorentz-frames
```

Create a virtual environment and install requirements

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

In case you want to run our experiments, you first have to collect the datasets. Small test datasets are contained in the `data/` directory. The full datasets can be downloaded from the Heidelberg ITP website ([amplitudes](https://www.thphys.uni-heidelberg.de/~plehn/data/amplitudes.hdf5), [toptagging](https://www.thphys.uni-heidelberg.de/~plehn/data/toptagging_full.npz), [event-generation](https://www.thphys.uni-heidelberg.de/~plehn/data/event_generation_ttbar.hdf5)) and from zenodo ([amplitudesXL](https://zenodo.org/records/16793011), [JetClass](https://zenodo.org/records/6619768), [TopTagXL](https://zenodo.org/records/10878355)). 
hdf5 archives have to be unpacked to npy files for each key in the archive. Finally, keys in the `data` section of the config files have to be adapted to specify where the datasets are located on your machine (`data_path` or `data_dir` depending on the experiment). The following command automates this procedure, and modifying the script allows you to collect only some datasets
```bash
python data/collect_data.py
```

<span style="color:red">xformers on MacOS</span> The LLoCa transformer taggers use xformers' `xformers.ops.memory_efficient_attention` as attention backend, because it supports block-diagonal attention matrices that allow us to save a factor of ~2 of RAM usage compared to standard torch attention with zero-padding for different-length jets. Unfortunately, [xformers does not support MacOS anymore](https://github.com/facebookresearch/xformers/issues/775).


## 2. Running tests

Most parts of the code are covered with unit tests. Before running any experiments, you can check that the code is healthy by running these tests

```bash
pytest tests
```

If they all pass you are good. If not, it either means that there is a problem with your environment, or that someone pushed changes that made the tests crash. In any case, you should get these tests to pass before moving on to running the main code.

## 3. Running experiments

You can run a quick test toptagging experiment with the following command

```bash
python run.py
```

We use hydra for configuration management, allowing to quickly override parameters in e.g. `config_quick/toptagging.yaml`. Configuration files for small test runs are in `config_quick`, if you want to run the big runs you should use `-cn config`. The `model`, `training`, `lframesnet` and `equivectors` option can be selected as follows, and individual keys can be modified with the `.` operator

```bash
python run.py -cp config -cn toptagging model=graphnet training=graphnet model/lframesnet=orthogonal model/lframesnet/equivectors=equigraph training.iterations=42
```

Further, we use mlflow for tracking. You can start a mlflow server based on the saved results in runs/tracking/mlflow.db on port 4242 of your machine with the following command

```bash
mlflow ui --port 4242 --backend-store-uri sqlite:///runs/tracking/mlflow.db
```

An existing run can be reloaded to perform additional tests with the trained model. For a previous run with exp_name=toptagging and run_name=hello_world_toptagging, one can run for example

```bash
python run.py -cn config -cp runs/toptagging/hello_world_toptagging train=false warm_start_idx=0
```

The warm_start_idx specifies which model in the models folder should be loaded and defaults to 0. 

## 4. Citation

If you find this code useful in your research, please cite our paper

```bibtex
@article{Spinner:2025prg,
    author = "Spinner, Jonas and Favaro, Luigi and Lippmann, Peter and Pitz, Sebastian and Gerhartz, Gerrit and Plehn, Tilman and Hamprecht, Fred A.",
    title = "{Lorentz Local Canonicalization: How to Make Any Network Lorentz-Equivariant}",
    eprint = "2505.20280",
    archivePrefix = "arXiv",
    primaryClass = "stat.ML",
    month = "5",
    year = "2025"
}
@article{Favaro:2025pgz,
    author = "Favaro, Luigi and Gerhartz, Gerrit and Hamprecht, Fred A. and Lippmann, Peter and Pitz, Sebastian and Plehn, Tilman and Qu, Huilin and Spinner, Jonas",
    title = "{Lorentz-Equivariance without Limitations}",
    eprint = "2508.14898",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "8",
    year = "2025"
}
```