import ray.tune as tune
import numpy as np
import wandb
from dotenv import dotenv_values, get_key
from pathlib import Path
import yaml


from .config import Config

def build_search_space(yaml_path):
    space = {}
    with open(yaml_path,"r") as f:
        yaml_dict = yaml.safe_load(f)
    for k, v in yaml_dict.items():
        t = v["type"]
        if t == "loguniform":
            space[k] = tune.loguniform(float(v["lower"]), float(v["upper"]))
        elif t == "uniform":
            space[k] = tune.uniform(float(v["lower"]), float(v["upper"]))
        elif t == "randint":
            space[k] = tune.randint(int(v["lower"]), int(v["upper"]))
        elif t == "choice":
            space[k] = tune.choice(v["values"])
        elif t == "sample_from":
            expr = v["expression"]
            # 安全地 eval 一个 lambda 表达式
            space[k] = tune.sample_from(lambda spec, e=expr: eval(e, {"np": np, "spec": spec}))
        elif t == "const":
            space[k] = v["value"]
        elif t == "grid_search":
            space[k] = tune.grid_search(v["values"])
        else:
            raise ValueError(f"Unsupported tune type: {t}")
    return space

def recursive_getattr(obj, attr_path):
    """ 辅助函数：递归获取属性对象 """
    for part in attr_path.split('.'):
        obj = getattr(obj, part)
    return obj

def recursive_setattr(obj, attr_path, value):
    """
    递归设置属性
    :param obj: 目标对象
    :param attr_path: 属性路径字符串，如 "env.project"
    :param value: 要设置的值
    """
    pre, _, post = attr_path.rpartition('.')
    # 如果有路径（如 a.b.c 中的 a.b），先递归获取 a.b 对象
    target = recursive_getattr(obj, pre) if pre else obj
    setattr(target, post, value)

def prefill_pipeline(pipeline,prefill_config):
    def prefilled_pipeline(config,*args,**kwargs):
        for k,v in config.items():
            recursive_setattr(prefill_config,k,v)
        pipeline(prefill_config,*args,**kwargs)
    return prefilled_pipeline

def wandb_log(func):
    def pipeline(config:Config,*args,**kwargs):
        wandb.login(
            key=get_key(".env","WANDB_API_KEY"),
            host=get_key(".env","WANDB_HOST")
        )
        wandb.init(
            project=config.env.project,
            name=config.env.name,
            notes=config.env.notes,
            tags=config.env.tags,
            config=config,
            group=config.env.group,
            job_type=config.env.job_type,
            mode=config.env.wandb_mode,
            save_code=True,
            config_exclude_keys=list(dotenv_values(".env").keys())+["tune.search_space"]
        )
        wandb.run.log_code(Path(__file__).resolve().parent)
        if config.tune.search_space:
            wandb.run.log_code(config.tune.search_space)
        func(config,*args,**kwargs)
        wandb.finish()
    return pipeline

