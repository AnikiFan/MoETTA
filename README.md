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
```

## Add Configuration

Base configuration is located at `config/config.py`.

Derived configuration can be stored in `config/subconfigs/` and `config/subconfigs/potpourri.py` serves as an example.

To add a configuration, only two things need to be done:

1. Add a configuration file into `config/subconfigs/`
2. Import the added file into `config/__init__.py`
