
## Reproducing the LLoCa results

This file provides the commands used to reproduce the results presented in our papers. Please note that these commands were written from memory and have not been re-tested. If you encounter any errors or inconsistencies, feel free to open a GitHub issue or contact us by email.

We hope this serves as a helpful resource for anyone interested in experimenting with the code and deepening their understanding of LLoCa. The repository includes many additional options and configurations beyond those discussed in the paper, and we have not yet had the opportunity to explore them all.

### 1) Setup environment

```bash
git clone https://github.com/heidelberg-hepml/lloca-experiments
cd lloca-experiments
```

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Quickly test your environment with the `config_quick/` configs. They support all options and are great to quickly get a feeling for what is going on with some manual print statements etc. These runs use small datasets shipped with the repo under `data/`.
```bash
pytest tests
python run_workflows.py
python run.py -cp config_quick -cn amplitudes save=false
python run.py -cp config_quick -cn toptagging save=false
python run.py -cp config_quick -cn ttbar save=false
```

<span style="color:red">xformers on MacOS</span> The Transformer, LLoCa-Transformer and L-GATr taggers use xformers' `memory_efficient_attention` as attention backend, because it supports block-diagonal attention matrices that allow us to save a factor of ~2 of RAM usage compared to standard torch attention with zero-padding for different-length jets. Unfortunately, [xformers does not support MacOS anymore](https://github.com/facebookresearch/xformers/issues/775). As a Mac user, we recommend to run this code on a HPC cluster in an interactive session. Note that LLoCa/L-GATr taggers can also be used without xformers, but that requires modifying the data embedding and attention mask construction. If you want to run just amplitude regression or event generation, it should be possible to just comment out the xformers imports in `experiments/`.

### 2) Collect datasets

- Amplitude regression: Download from https://zenodo.org/records/16793011; set `data.data_path` in `config/amplitudesxl.yaml`
- Event generation: Run `python data/collect_data.py eventgen`; should have files `data/ttbar_nj.npy` with n=0,1,2,3,4
- Top-tagging: Run `python data/collect_data.py toptagging`; should have file `data/toptagging_full.npz`
- JetClass: Download from https://zenodo.org/records/6619768; set `data.data_dir` in `config/jctagging.yaml`
- TopTagXL: Download from https://zenodo.org/records/10878355; set `data.data_dir` in `config/toptagxl.yaml`

### 3) Amplitude regression

https://arxiv.org/abs/2508.14898 Figure 2 / https://arxiv.org/abs/2505.20280 Figure 3. 'Memory' is printed in the log file as `GPU_RAM_max_used`.
```bash
python run.py -cp config -cn amplitudesxl data.num_train_files=10 data.data_path=data/zgggg_10M data.dataset=zgggg_0 model=amp_dsi
python run.py -cp config -cn amplitudesxl data.num_train_files=10 data.data_path=data/zgggg_10M data.dataset=zgggg_0 model=amp_lgatr
python run.py -cp config -cn amplitudesxl data.num_train_files=10 data.data_path=data/zgggg_10M data.dataset=zgggg_0 model=amp_graphnet
python run.py -cp config -cn amplitudesxl data.num_train_files=10 data.data_path=data/zgggg_10M data.dataset=zgggg_0 model=amp_graphnet model/framesnet=learnedpd
python run.py -cp config -cn amplitudesxl data.num_train_files=10 data.data_path=data/zgggg_10M data.dataset=zgggg_0 model=amp_transformer
python run.py -cp config -cn amplitudesxl data.num_train_files=10 data.data_path=data/zgggg_10M data.dataset=zgggg_0 model=amp_transformer model/framesnet=learnedpd
# repeat for zg, zgg, zggg

pytest tests/experiments/test_amp_flops.py -s #FLOPs
# Time evaluated on 1xH100
```

https://arxiv.org/abs/2508.14898 Figure 3 / https://arxiv.org/abs/2505.20280 Figure 4
```bash
python run.py -cp config -cn amplitudesxl data.num_train_files=1 data.subsample=1000 model=amp_dsi
python run.py -cp config -cn amplitudesxl data.num_train_files=1 data.subsample=10000 model=amp_dsi
python run.py -cp config -cn amplitudesxl data.num_train_files=1 data.subsample=100000 model=amp_dsi
python run.py -cp config -cn amplitudesxl data.num_train_files=1 model=amp_dsi
python run.py -cp config -cn amplitudesxl data.num_train_files=10 model=amp_dsi
python run.py -cp config -cn amplitudesxl data.num_train_files=100 model=amp_dsi
# Repeat for the different models by changing model=... and model/framesnet=... like above
```

https://arxiv.org/abs/2508.14898 Figure 4 / https://arxiv.org/abs/2505.20280 Figure 6
```bash
# Left:
python run.py -cp config -cn amplitudesxl
python run.py -cp config -cn amplitudesxl model/framesnet=learnedso3
python run.py -cp config -cn amplitudesxl model/framesnet=learnedpd

# Right:
python run.py -cp config -cn amplitudesxl
python run.py -cp config -cn amplitudesxl model/framesnet=learnedpd model.framesnet.equivectors.hidden_channels=16
python run.py -cp config -cn amplitudesxl model/framesnet=learnedpd model.framesnet.fix_params=true
python run.py -cp config -cn amplitudesxl model/framesnet=learnedpd model.framesnet.equivectors.dropout_prob=0.2
python run.py -cp config -cn amplitudesxl model/framesnet=learnedpd model.framesnet.fix_params=true model.framesnet.equivectors.dropout_prob=0.2
python run.py -cp config -cn amplitudesxl model/framesnet=learnedpd

# For both, repeat for different dataset sizes like above
```

https://arxiv.org/abs/2508.14898 Table 1 / https://arxiv.org/abs/2505.20280 Table 2
```bash
# table rows one by one
python run.py -cp config -cn amplitudesxl
python run.py -cp config -cn amplitudesxl model/framesnet=learnedpd model.framesnet.is_global=true
python run.py -cp config -cn amplitudesxl model/framesnet=learnedpd model.net.attn_reps=16x0n
python run.py -cp config -cn amplitudesxl model/framesnet=learnedpd model.net.attn_reps=1x2n
python run.py -cp config -cn amplitudesxl model/framesnet=learnedpd model.net.attn_reps=4x1n
python run.py -cp config -cn amplitudesxl model/framesnet=learnedpd model.net.attn_reps=12x0n+1x1n
python run.py -cp config -cn amplitudesxl model/framesnet=learnedpd
```

https://arxiv.org/abs/2508.14898 Table 8
```bash
python run.py -cp config -cn amplitudesxl model=amp_transformer
python run.py -cp config -cn amplitudesxl model=amp_transformer model/framesnet=learnedpd model/framesnet/equivectors=lgatr
python run.py -cp config -cn amplitudesxl model=amp_transformer model/framesnet=learnedpd model/framesnet/equivectors=pelican
python run.py -cp config -cn amplitudesxl model=amp_transformer model/framesnet=learnedpd
```

https://arxiv.org/abs/2505.20280 Figure 5
```bash
# Left:
python run.py -cp config -cn amplitudesxl model=amp_mlp
python run.py -cp config -cn amplitudesxl model=amp_mlp model/framesnet=randomlorentz
python run.py -cp config -cn amplitudesxl model=amp_mlp model/framesnet=learnedpd

# Right:
python run.py -cp config -cn amplitudesxl model=amp_graphnet model.include_edges=false
python run.py -cp config -cn amplitudesxl model=amp_graphnet model.include_edges=false model/framesnet=randomlorentz
python run.py -cp config -cn amplitudesxl model=amp_graphnet model.include_edges=false model/framesnet=learnedpd

# For both, repeat for different dataset sizes like above
```

### 4) Event generation

https://arxiv.org/abs/2508.14898 Figure 5
```bash
python run.py -cp config -cn ttbar model=eg_transformer
python run.py -cp config -cn ttbar model=eg_lgatr
python run.py -cp config -cn ttbar model=eg_transformer model/framesnet=learnedpd
```

https://arxiv.org/abs/2508.14898 Figure 6
```bash
# Transformer top left
python run.py -cp config -cn ttbar data.train_test_val=[0.5,0.2,0.1]
python run.py -cp config -cn ttbar data.train_test_val=[0.2,0.2,0.1]
python run.py -cp config -cn ttbar data.train_test_val=[0.05,0.2,0.1]
python run.py -cp config -cn ttbar data.train_test_val=[0.02,0.2,0.1]
python run.py -cp config -cn ttbar data.train_test_val=[0.005,0.2,0.1]
python run.py -cp config -cn ttbar data.train_test_val=[0.002,0.2,0.1]
python run.py -cp config -cn ttbar data.train_test_val=[0.0005,0.2,0.1]
python run.py -cp config -cn ttbar data.train_test_val=[0.0002,0.2,0.1]

# Transformer top right
python run.py -cp config -cn ttbar data.n_jets=0
python run.py -cp config -cn ttbar data.n_jets=1
python run.py -cp config -cn ttbar data.n_jets=2
python run.py -cp config -cn ttbar data.n_jets=3
python run.py -cp config -cn ttbar data.n_jets=4

# Repeat each of these for different dataset sizes and multiplicities like above:
python run.py -cp config -cn ttbar model=eg_transformer model/framesnet=learnedpd
python run.py -cp config -cn ttbar model=eg_transformer model/framesnet=randomxyrotation # only upper two plots
python run.py -cp config -cn ttbar model=eg_lgatr # only upper two plots
python run.py -cp config -cn ttbar model=eg_transformer model/framesnet=learnedso2 # only lower two plots
python run.py -cp config -cn ttbar model=eg_transformer model/framesnet=learnedso3 # only lower two plots
```

https://arxiv.org/abs/2508.14898 Table 2
```bash
python run.py -cp config -cn ttbar model=eg_transformer
python run.py -cp config -cn ttbar model=eg_transformer model/framesnet=learnedpd model.net.attn_reps=13x0n
python run.py -cp config -cn ttbar model=eg_transformer model/framesnet=learnedpd model.net.attn_reps=9x0n+1x1n
python run.py -cp config -cn ttbar model=eg_transformer model/framesnet=learnedpd model.net.attn_reps=5x0n+2x1n
```

### 5) Jet tagging

https://arxiv.org/abs/2508.14898 Table 3 and Table 9 / https://arxiv.org/abs/2505.20280 Table 1
```bash
# For PFN and P-CNN we used https://github.com/jet-universe/particle_transformer; FLOPs and memory were extracted manually
python run.py -cp config -cn jctagging model=tag_MIParT-L
python run.py -cp config -cn jctagging model=tag_lorentznet
python run.py -cp config -cn jctagging model=tag_pelican_fair
python run.py -cp config -cn jctagging training=jc_lgatr model=tag_lgatr
python run.py -cp config -cn jctagging model=tag_particlenet
python run.py -cp config -cn jctagging model=tag_particlenet model/framesnet=learnedpd
python run.py -cp config -cn jctagging model=tag_ParT
python run.py -cp config -cn jctagging model=tag_ParT model/framesnet=learnedpd
python run.py -cp config -cn jctagging model=tag_transformer
python run.py -cp config -cn jctagging model=tag_transformer model/framesnet=learnedpd

# extra entries in Table 9
python run.py -cp config -cn jctagging model=tag_transformer model/framesnet=learnedpd model/framesnet/equivectors=lgatr
python run.py -cp config -cn jctagging model=tag_transformer model/framesnet=learnedpd model/framesnet/equivectors=pelican

pytest tests/experiments/test_tag_flops.py -s # FLOPs

# Commands of this form were used to estimate the timing based on 1k iterations. We find that these estimates deviate slightly from the full jetclass trainings above.
python run.py -cp config -cn toptagging model=tag_transformer training.iterations=1000
```

https://arxiv.org/abs/2508.14898 Table 4
```bash
python run.py -cp config -cn jctagging model=tag_transformer
python run.py -cp config -cn jctagging model=tag_ParT
python run.py -cp config -cn jctagging model=tag_transformer model/framesnet=learnedpd model.framesnet.is_global=true
python run.py -cp config -cn jctagging model=tag_transformer model/framesnet=learnedpd model.net.attn_reps=16x0n
python run.py -cp config -cn jctagging model=tag_transformer model/framesnet=learnedpd model.net.attn_reps=4x1n
python run.py -cp config -cn jctagging model=tag_transformer model/framesnet=learnedpd model.net.attn_reps=8x0n+2x1n
python run.py -cp config -cn jctagging model=tag_transformer model/framesnet=learnedpd model.net.attn_reps=12x0n+1x1n
```

https://arxiv.org/abs/2508.14898 Table 5
```bash
# everything except the last two entries are taken from the literature
python run.py -cp config -cn toptagging model=tag_top_transformer
python run.py -cp config -cn toptagging model=tag_top_transformer model/framesnet=learnedpd model.framesnet.equivectors.dropout_prob=0.1
```

https://arxiv.org/abs/2508.14898 Table 6
```bash
python run.py -cp config -cn toptagxl model=tag_MIParT-L
python run.py -cp config -cn toptagxl model=tag_lorentznet
python run.py -cp config -cn toptagxl model=tag_lgatr
python run.py -cp config -cn toptagxl model=tag_particlenet
python run.py -cp config -cn toptagxl model=tag_particlenet model/framesnet=learnedpd
python run.py -cp config -cn toptagxl model=tag_ParT
python run.py -cp config -cn toptagxl model=tag_ParTmodel/framesnet=learnedpd
python run.py -cp config -cn toptagxl model=tag_transformer
python run.py -cp config -cn toptagxl model=tag_transformer model/framesnet=learnedpd
```

https://arxiv.org/abs/2508.14898 Table 7
```bash
# Architecture-level symmetry breaking
python run.py -cp config -cn jctagging data.beam_reference=null data.add_time_reference=false # non-equivariant
python run.py -cp config -cn jctagging data.beam_reference=null data.add_time_reference=false model/framesnet=learnedso2 # SO(2)
python run.py -cp config -cn jctagging data.tagging_features_framesnet=zinvariant data.beam_reference=null data.add_time_reference=false model/framesnet=learnedz # SO(1,1)xSO(2)
python run.py -cp config -cn jctagging data.tagging_features_framesnet=so3invariant data.beam_reference=null data.add_time_reference=false model/framesnet=learnedso3 # SO(3)
python run.py -cp config -cn jctagging data.tagging_features_framesnet=null data.beam_reference=null data.add_time_reference=false model/framesnet=learnedpd # SO(1,3)

# Input-level symmetry breaking
python run.py -cp config -cn jctagging model/framesnet=learnedpd data.beam_reference=all # non-equivariant
python run.py -cp config -cn jctagging model/framesnet=learnedpd # SO(2)
python run.py -cp config -cn jctagging model/framesnet=learnedpd data.tagging_features_framesnet=zinvariant data.add_time_reference=false # SO(1,1)xSO(2)
python run.py -cp config -cn jctagging model/framesnet=learnedpd data.tagging_features_framesnet=so3invariant data.beam_reference=null # SO(3)
# SO(1,3) is the same as for 'architecture'
```

### 6) Extra features in the code

- Warm-start runs for evaluation or continue-training (use `training.scheduler_scale`) by pointing `-cp` and `-cn` to an existing experiment and use `warm_start_idx` to select the run to load
- Tracking with `mlflow` (set `use_mlflow=true`)
- ML tricks that we don't use by default (`amp`, `float32_matmul_precision`, `ema`)
