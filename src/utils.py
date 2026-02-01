import ray.tune as tune
import numpy as np
import wandb
from dotenv import dotenv_values, get_key
from pathlib import Path
import torch
import yaml
import random
import functools
from loguru import logger
import time
from pprint import pformat
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


from config import Config


def build_search_space(yaml_path):
    space = {}
    with open(yaml_path, "r") as f:
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
            space[k] = tune.sample_from(
                lambda spec, e=expr: eval(e, {"np": np, "spec": spec})
            )
        elif t == "const":
            space[k] = v["value"]
        elif t == "grid_search":
            space[k] = tune.grid_search(v["values"])
        else:
            raise ValueError(f"Unsupported tune type: {t}")
    return space


def recursive_getattr(obj, attr_path):
    """辅助函数：递归获取属性对象"""
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    return obj


def recursive_setattr(obj, attr_path, value):
    """
    递归设置属性
    :param obj: 目标对象
    :param attr_path: 属性路径字符串，如 "env.project"
    :param value: 要设置的值
    """
    pre, _, post = attr_path.rpartition(".")
    # 如果有路径（如 a.b.c 中的 a.b），先递归获取 a.b 对象
    target = recursive_getattr(obj, pre) if pre else obj
    setattr(target, post, value)


def prefill_pipeline(pipeline, prefill_config):
    @functools.wraps(pipeline)
    def prefilled_pipeline(config, *args, **kwargs):
        for k, v in config.items():
            recursive_setattr(prefill_config, k, v)
        return pipeline(prefill_config, *args, **kwargs)

    return prefilled_pipeline


def wandb_log(func):
    @functools.wraps(func)
    def pipeline(config: Config, *args, **kwargs):
        wandb.login(
            key=get_key(".env", "WANDB_API_KEY"), host=get_key(".env", "WANDB_BASE_URL")
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
            config_exclude_keys=list(dotenv_values(".env").keys())
            + ["tune.search_space"], # exclude sensitive info and large files
        )
        wandb.run.log_code(Path(__file__).resolve().parent)
        if config.tune.search_space:
            wandb.run.log_code(config.tune.search_space)
        output = func(config, *args, **kwargs)
        time.sleep(1) # ensure all logs are flushed
        wandb.finish()
        return output

    return pipeline


def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"Set seed to {seed}")


def deterministic(function):
    @functools.wraps(function)
    def deterministic_function(config: Config, *args, **kwargs):
        set_random_seed(seed=config.train.seed)
        return function(config, *args, **kwargs)

    return deterministic_function


def show_config(func):
    @functools.wraps(func)
    def function(config, *args, **kwargs):
        logger.info(f"""Start function {func.__name__} with config:
                    {pformat(config)}""")
        return func(config, *args, **kwargs)

    return function


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info(f"Function [{func.__name__}] cost: {duration:.4f} s")
        try:
            wandb.summary[f"{func.__name__}_duration"] = duration
        except Exception as e:
            logger.warning(f"Failed to log duration to wandb: {e}")
        return result

    return wrapper


def mem_trace(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            return func(*args, **kwargs)

        # 清空显存缓存，确保统计的是当前函数的纯增量
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        initial_mem = torch.cuda.memory_allocated()

        result = func(*args, **kwargs)
        # 必须同步，否则计时/计显存不准
        torch.cuda.synchronize()

        peak_mem = torch.cuda.max_memory_allocated()
        final_mem = torch.cuda.memory_allocated()

        logger.info(f"--- [{func.__name__}] VRAM Usage Report ---")
        logger.info(f"  Peak VRAM Usage: {peak_mem / 1024**2:.2f} MB")
        logger.info(
            f"  Net VRAM Increase: {(final_mem - initial_mem) / 1024**2:.2f} MB"
        )
        logger.info("--------------------------")
        wandb.summary[f"{func.__name__}_peak_VRAM_usage"] = peak_mem / 1024**2
        wandb.summary[f"{func.__name__}_net_VRAM_increase"] = (
            final_mem - initial_mem
        ) / 1024**2
        return result

    return wrapper


class CumulativeTimer:
    def __init__(self, func):
        functools.wraps(func)(self)
        self.func = func
        self.total_time = 0.0
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        start_time = time.perf_counter()

        result = self.func(*args, **kwargs)

        duration = time.perf_counter() - start_time
        self.total_time += duration
        self.call_count += 1

        # 实时打印本次耗时
        logger.debug(
            f"[{self.func.__name__}] Call #{self.call_count} finished in {duration:.4f}s"
        )
        logger.debug(self.stats())
        wandb.summary[f"{self.func.__name__}_total_calls"] = self.call_count
        wandb.summary[f"{self.func.__name__}_total_time"] = self.total_time
        wandb.summary[f"{self.func.__name__}_avg_time"] = (
            self.total_time / self.call_count if self.call_count > 0 else 0
        )
        return result

    def stats(self):
        """返回统计信息的格式化字符串"""
        return (
            f"\n{'=' * 30}\n"
            f"Cumulative Stats for [{self.func.__name__}]:\n"
            f"  Total Calls: {self.call_count}\n"
            f"  Total Time:  {self.total_time:.4f}s\n"
            f"  Avg Time:    {self.total_time / self.call_count if self.call_count > 0 else 0:.4f}s\n"
            f"{'=' * 30}"
        )

    def __del__(self):
        """当脚本结束或对象销毁时自动打印最终统计"""
        if self.call_count > 0:
            # 注意：在某些复杂的分布式环境下，__del__ 的触发时机可能不确定
            # 建议在 main 结束前手动调用 .stats()
            logger.info(self.stats())


def count_correct(output, target, topk=(1,)):
    """
    Computes the number of correct predictions for the specified values of topk.

    Args:
        output (Tensor): Model outputs, shape [batch_size, num_classes]
        target (Tensor): Ground truth labels, shape [batch_size]
        topk (tuple): Tuple of integers specifying which top-k accuracies to compute.

    Returns:
        A list with the number of correct predictions for each k in topk.
    """
    maxk = max(topk)

    # topk: get top maxk predicted classes for each sample
    _, indices = output.topk(
        maxk, dim=1, largest=True, sorted=True
    )  # [batch_size, maxk]
    indices = indices.t()  # [maxk, batch_size]
    correct = indices.eq(target.view(1, -1).expand_as(indices))  # [maxk, batch_size]

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)  # count of correct top-k
        res.append(int(correct_k.item()))  # return as int
    return res


def get_logger(config:Config):
    @torch.no_grad()
    def logger(locals):
        self = locals["self"]
        if self.step % config.algo.moetta.log_matrix_step == 0:
            if self.self_router:
                df = pd.DataFrame(
                    {
                        "no": [f"expert{i}" for i in np.arange(self.num_expert)],
                        "value": self.cnt.cpu(),
                    }
                )
                plt.figure(figsize=(6, 4))
                ax = sns.barplot(x="no", y="value", data=df)
                plt.title("Expert Distribution")
                plt.tight_layout()
                wandb.log(
                    {f"expert_distribution/layer{self.idx}": wandb.Image(plt)},
                    step=self.step,
                )
                plt.close()

            # expert weight cosine similarity
            weight = self.experts_weight.detach()
            bias = self.experts_bias.detach()
            weight = weight / weight.norm(p=2, dim=-1).unsqueeze(-1)
            bias = bias / bias.norm(p=2, dim=-1).unsqueeze(-1)

            weight_cosine_similarity = torch.matmul(weight, weight.T)
            fig, ax = plt.subplots()
            cax = ax.imshow(
                weight_cosine_similarity.cpu(), cmap="hot", interpolation="nearest"
            )
            fig.colorbar(cax)

            log_dict = {
                f"expert_weight_cosine_similarity/layer{self.idx}": wandb.Image(fig),
                f"weight_cosine_similarity_mean/layer{self.idx}": lower_triangle_mean(
                    weight_cosine_similarity
                ),
            }
            wandb.log(log_dict, step=self.step)
            plt.close()

            bias_cosine_similarity = torch.matmul(bias, bias.T)
            fig, ax = plt.subplots()
            cax = ax.imshow(
                bias_cosine_similarity.cpu(), cmap="hot", interpolation="nearest"
            )
            fig.colorbar(cax)

            log_dict = {
                f"expert_bias_cosine_similarity/layer{self.idx}": wandb.Image(fig),
                f"bias_cosine_similarity_mean/layer{self.idx}": lower_triangle_mean(
                    bias_cosine_similarity
                ),
            }
            wandb.log(log_dict, step=self.step)
            plt.close()
    return logger


def set_nested_attr(root, name: str, value):
    parts = name.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def cosine_similarity(tensor_list):
    """
    计算 list 中所有 1D tensor 之间的 cosine similarity

    参数:
        tensor_list (list of torch.Tensor): 形状相同的 1D tensor 列表

    返回:
        torch.Tensor: (N, N) 形状的对称矩阵，表示两两之间的 cosine similarity
    """
    # 将 list 转换为 2D 矩阵 (N, D)，N 是 tensor 数量，D 是每个 tensor 的长度
    stacked_tensors = torch.stack(tensor_list)  # (N, D)

    # 计算余弦相似度 (cosine similarity) 矩阵
    similarity_matrix = torch.nn.functional.cosine_similarity(
        stacked_tensors.unsqueeze(1),  # (N, 1, D)
        stacked_tensors.unsqueeze(0),  # (1, N, D)
        dim=2,  # 在特征维度计算
    )  # 得到 (N, N) 形状的相似度矩阵

    return similarity_matrix


def symmetric_kl_divergence(mean_list, var_list):
    """
    计算对称 KL 散度矩阵，并归一化

    参数:
        mean_list (list of torch.Tensor): 每个元素是 1D tensor，表示正态分布的均值 (mu)
        var_list (list of torch.Tensor): 每个元素是 1D tensor，表示正态分布的方差 (sigma^2)

    返回:
        torch.Tensor: (N, N) 形状的对称 KL 散度矩阵（已归一化）
    """
    # 转换为张量 (N, D)
    means = torch.stack(mean_list)  # (N, D)
    vars = torch.stack(var_list)  # (N, D)

    N, D = means.shape  # N 是分布个数，D 是向量长度

    # 扩展维度进行广播计算 (N, 1, D) 和 (1, N, D) 变为 (N, N, D)
    mu_p = means.unsqueeze(1)  # (N, 1, D)
    mu_q = means.unsqueeze(0)  # (1, N, D)
    var_p = vars.unsqueeze(1)  # (N, 1, D)
    var_q = vars.unsqueeze(0)  # (1, N, D)

    # 计算 KL(p || q)
    kl_pq = 0.5 * (
        (var_q / var_p) + ((mu_p - mu_q) ** 2) / var_p - 1 + torch.log(var_p / var_q)
    ).sum(dim=2)

    # 计算 KL(q || p)
    kl_qp = 0.5 * (
        (var_p / var_q) + ((mu_q - mu_p) ** 2) / var_q - 1 + torch.log(var_q / var_p)
    ).sum(dim=2)

    # 计算对称 KL 散度
    sym_kl_matrix = 0.5 * (kl_pq + kl_qp)

    # 归一化 (除以 D)
    sym_kl_matrix /= D

    return sym_kl_matrix


def lower_triangle_mean(mat: torch.Tensor) -> torch.Tensor:
    """
    计算一个方阵的下三角区域（不含主对角线）的平均值

    参数:
        mat (torch.Tensor): 一个二维方阵张量，形状为 (n, n)

    返回:
        torch.Tensor: 下三角区域的平均值（标量张量）
    """
    if mat.ndim != 2 or mat.size(0) != mat.size(1):
        raise ValueError("Input must be a 2D square matrix.")

    # 创建下三角 mask（不含主对角线）
    mask = torch.tril(torch.ones_like(mat, dtype=torch.bool), diagonal=-1)

    # 提取下三角元素并求平均
    return mat[mask].mean()


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
