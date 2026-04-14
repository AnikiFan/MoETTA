# MoETTA

## Set Up Environment

1. Run following command to set up codebase and Python environment.

```bash
git clone https://github.com/AnikiFan/MoETTA.git
cd MoETTA
# In case you haven't install uv, run following command if you are using Linux
# curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run ray start --head
```

2. Create `.env` file under `MoETTA` directory.

```bash
RAY_ADDRESS=<YOUR RAY SERVER ADDRESS> # Get it by running `uv run ray start --head`
WANDB_API_KEY=<YOUR WANDB API KEY>
WANDB_BASE_URL=<YOUR WANDB SERVER URL> # If you are not using wandb-local, then fill it with `https://api.wandb.ai`
```

3. Tailor the environment configuration to yours.

The base configuration is located at `config/config.py`, where the configuration related to path needed to be changed according to your environment.

### CIFAR / CIFAR-C Support

The codebase now supports:

- clean `CIFAR-10` and `CIFAR-100` through `--data.dataset cifar10|cifar100 --data.corruption original`
- mixed-corruption `CIFAR-10-C` and `CIFAR-100-C` through `--data.corruption cifar10-c|cifar100-c`
- single CIFAR-C corruption through `--data.dataset cifar10|cifar100 --data.corruption gaussian_noise` and other common corruption names

Please update the following paths in `config/config.py` to match your environment:

- `env.cifar10_data_path`
- `env.cifar100_data_path`
- `env.cifar10_c_path`
- `env.cifar100_c_path`

If you evaluate a CIFAR model, it is strongly recommended to provide a CIFAR-trained checkpoint:

```bash
uv run main.py base \
  --data.dataset cifar10 \
  --data.corruption original \
  --data.num_class 10 \
  --model.model resnet18 \
  --model.pretrained false \
  --model.checkpoint_path /path/to/cifar10_checkpoint.pth
```

You can also let the code download a Hugging Face `timm` checkpoint automatically:

```bash
uv run main.py base \
  --data.corruption cifar10-c \
  --model.model resnet18 \
  --model.hf_repo_id SamAdamDay/resnet18_cifar10
```

For CIFAR-C, the expected extracted directory layout is:

```text
<cifar10_c_path>/
  labels.npy
  gaussian_noise.npy
  shot_noise.npy
  impulse_noise.npy
  ...
```

## Run Experiment

```base
# Run an experiment locally, i.e., without ray
uv run main.py base --env.local

# Run an experiment with wandb offline
uv run main.py base --env.wandb_mode offline

# Run a hyper-parameter tuning/sweep by designating search space configuration
uv run main.py base --tune.search_space /home/fx25/workspace/MoETTA/config/search_space/seed.yaml

uv run main.py base --algo.algorithm eata

uv run main.py base --algo.algorithm moetta

uv run main.py base --algo.algorithm moetta --data.corruption potpourri+

uv run main.py base --algo.algorithm tent --data.corruption cifar10-c --model.model resnet18 --model.pretrained false --model.checkpoint_path /path/to/cifar10_checkpoint.pth

uv run main.py base --algo.algorithm tent --data.dataset cifar100 --data.corruption gaussian_noise --data.num_class 100 --model.model resnet18 --model.pretrained false --model.checkpoint_path /path/to/cifar100_checkpoint.pth
```

## Add Configuration

Base configuration is located at `config/config.py`.

Derived configuration can be stored in `config/subconfigs/` and `config/subconfigs/potpourri.py` serves as an example.

To add a configuration, only two things need to be done:

1. Add a configuration file into `config/subconfigs/`
2. Import the added file into `config/__init__.py`

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/FanJCHCJZTW26,
  author       = {Xiao Fan and
                  Jingyan Jiang and
                  Zhaoru Chen and
                  Fanding Huang and
                  Xiao Chen and
                  Qinting Jiang and
                  Bowen Zhang and
                  Xing Tang and
                  Zhi Wang},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {MoETTA: Test-Time Adaptation Under Mixed Distribution Shifts with
                  MoE-LayerNorm},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {21011--21019},
  publisher    = {{AAAI} Press},
  year         = {2026},
  url          = {https://doi.org/10.1609/aaai.v40i25.39243},
  doi          = {10.1609/AAAI.V40I25.39243},
  timestamp    = {Fri, 27 Mar 2026 17:13:39 +0100},
  biburl       = {https://dblp.org/rec/conf/aaai/FanJCHCJZTW26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
