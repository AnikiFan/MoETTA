import tyro
import yaml
import ray.tune.schedulers as schedulers
import ray.tune as tune
import ray
from ray.runtime_env import RuntimeEnv
from pathlib import Path
from dotenv import get_key


from src.config import Config
from src.pipeline import pipeline
from src.utils import build_search_space, prefill_pipeline

def main():
    config = tyro.cli(Config)
    if config.env.local:
        pipeline(config)
        return
    ray.init(
        address=get_key(".env","RAY_ADDRESS"),
        runtime_env=RuntimeEnv(
            env_vars=dict(
                WANDB_API_KEY=get_key(".env","WANDB_API_KEY"),
                UV_LINK_MODE="copy",
                VIRTUAL_ENV=".venv"
            ),
            working_dir=str(Path(__file__).resolve().parent),
            # https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments
            excludes=[
                "/data/",
                "/.venv/",
                "/__pycache__/",
            ]
        )
    )
    if not config.tune.search_space:
        ray.get(ray.remote(pipeline).remote(config))
    else:
        Scheduler = getattr(schedulers, config.tune.scheduler)
        scheduler = Scheduler(max_t=config.tune.max_t)
        prefilled_pipeline = prefill_pipeline(pipeline,config)
        pipeline_with_resources = tune.with_resources(
            prefilled_pipeline,
            dict(cpu=config.tune.cpu_per_trial,gpu=config.tune.gpu_per_trial)
        )
        search_space = build_search_space(config.tune.search_space)
        tune_pipeline = tune.Tuner(
            pipeline_with_resources,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                mode=config.tune.mode,
                metric=config.tune.metric,
                num_samples=config.tune.num_samples,
                search_alg=config.tune.search_algorithm,
                scheduler=scheduler,
            ),
            run_config=tune.RunConfig(
            )
        )
        tune_pipeline.fit()

    ray.shutdown()



if __name__ == "__main__":
    main()