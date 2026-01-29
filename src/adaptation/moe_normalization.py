import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Tuple
import wandb

from ..config import Config
from ..utils import set_nested_attr


@torch.jit.script
def fuse_params(
    experts_weight: torch.Tensor,
    experts_bias: torch.Tensor,
    topks: torch.Tensor,
    coeff: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = topks.size(0)
    d = experts_weight.size(1)

    expanded_experts_weight = experts_weight.unsqueeze(0).expand(B, -1, -1)
    expanded_experts_bias = experts_bias.unsqueeze(0).expand(B, -1, -1)
    expanded_topks = topks.unsqueeze(-1).expand(-1, -1, d)

    selected_weight = expanded_experts_weight.gather(1, expanded_topks)
    selected_bias = expanded_experts_bias.gather(1, expanded_topks)

    selected_coeff = coeff.gather(1, topks).unsqueeze(-1)

    fused_weight = (selected_weight * selected_coeff).sum(1)
    fused_bias = (selected_bias * selected_coeff).sum(1)

    final_weight = fused_weight + shared_weight.unsqueeze(0)
    final_bias = fused_bias + shared_bias.unsqueeze(0)

    return final_weight, final_bias


@torch.jit.script
def apply_layernorm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    if x.dim() == 3:
        # [B, N, D]
        return x_norm * weight.unsqueeze(1) + bias.unsqueeze(1)
    elif x.dim() == 4:
        # [B, H, W, D]
        return x_norm * weight.unsqueeze(1).unsqueeze(1) + bias.unsqueeze(1).unsqueeze(
            1
        )
    else:
        raise NotImplementedError


@torch.jit.script
def apply_batchnorm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    B, C, H, W = x.shape
    # 按 (B, C, H, W) → (B, C, H*W)，然后在(B, HW)上做均值/方差
    x_flat = x.permute(1, 0, 2, 3).contiguous().view(C, -1)

    mean = x_flat.mean(dim=1)
    var = x_flat.var(dim=1, unbiased=False)

    mean = mean.view(1, C, 1, 1)
    var = var.view(1, C, 1, 1)

    x_norm = (x - mean) / torch.sqrt(var + eps)

    weight = weight.view(B, C, 1, 1)
    bias = bias.view(B, C, 1, 1)

    out = x_norm * weight + bias
    return out


@torch.jit.script
def apply_groupnorm(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    B, C, H, W = x.shape
    G = num_groups
    assert C % G == 0, "channels must be divisible by num_groups"

    x = x.view(B, G, -1)

    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)

    x_norm = (x - mean) / torch.sqrt(var + eps)
    x_norm = x_norm.view(B, C, H, W)

    weight = weight.view(B, C, 1, 1)
    bias = bias.view(B, C, 1, 1)

    out = x_norm * weight + bias
    return out


class MoENormalizationLayer(nn.Module):
    """
    通用 MoE 归一化层基类，封装专家参数、路由器以及参数融合逻辑。
    可根据 base_mod 类型自动调用对应的归一化实现。
    """

    def __init__(
        self,
        idx: int,
        num_expert: int,
        shared_expert: bool,
        base_mod: nn.Module,
        randomness: float,
        self_router: bool = False,
        samplewise: bool = None,
        topk: int = None,
        weight_by_prob: bool = None,
        penalty: float = None,
        decay: float = None,
        logger: Callable = None,
        grad_hook: bool = False,
        pass_through_coeff: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        assert num_expert > 0, "专家数必须大于 0"
        self.idx = idx
        self.num_expert = num_expert
        self.base_mod = base_mod
        self.self_router = self_router

        # 从传入的 base_mod 提取基础参数
        self.weight = base_mod.weight
        self.weight.requires_grad_(shared_expert)
        self.bias = base_mod.bias
        self.bias.requires_grad_(shared_expert)

        # 初始化专家参数
        weight_shape = self.weight.shape
        bias_shape = self.bias.shape
        self.experts_weight = nn.Parameter(
            torch.randn((num_expert,) + tuple(weight_shape), device=device)
        )
        self.experts_bias = nn.Parameter(
            torch.randn((num_expert,) + tuple(bias_shape), device=device)
        )
        # 缩放专家参数
        w_norm = torch.norm(self.weight).item()
        b_norm = torch.norm(self.bias).item()
        expert_w_norm = torch.norm(self.experts_weight.view(num_expert, -1), dim=1)
        expert_b_norm = torch.norm(self.experts_bias.view(num_expert, -1), dim=1)
        factor_w = (randomness * w_norm / expert_w_norm).view(
            num_expert, *([1] * len(weight_shape))
        )
        factor_b = (randomness * b_norm / expert_b_norm).view(
            num_expert, *([1] * len(bias_shape))
        )
        self.experts_weight.data *= factor_w
        self.experts_bias.data *= factor_b

        # 路由器
        if self_router:
            in_features = self.weight.numel()
            self.router = nn.Linear(in_features, num_expert, device=device)
            nn.init.xavier_normal_(self.router.weight)
            nn.init.zeros_(self.router.bias)

        # 路由配置
        self.samplewise = samplewise
        self.topk = topk
        self.weight_by_prob = weight_by_prob
        self.penalty = torch.zeros(num_expert, device=device)
        self.route_penalty = penalty
        self.decay = decay
        self.step = 0
        self.coeff = None
        self.topks = None
        self.logger = logger
        self.pass_through_coeff = pass_through_coeff
        self.cnt = torch.zeros(num_expert, device=device)
        self.lb_loss = None
        self.route_prob_list = []
        self.cls_feature = None

        # 可选梯度钩子注册略
        if grad_hook:
            hook_fn = self.make_step_aware_hook(f"layer{self.idx}", self)
            self.register_expert_block_hook(
                self.experts_weight, kind="weight", idx=self.idx, hook_fn=hook_fn
            )
            self.register_expert_block_hook(
                self.experts_bias, kind="bias", idx=self.idx, hook_fn=hook_fn
            )
            if self.router is not None:
                self.register_router_separate_hooks(
                    self.router, f"layer{self.idx}/router", hook_fn
                )

    @staticmethod
    def make_step_aware_hook(name_prefix, instance):
        def hook_fn(name, norm_tensor):
            full_name = f"{name_prefix}/{name}"
            wandb.log({full_name: norm_tensor.item()}, step=instance.step)

        return hook_fn

    @staticmethod
    def register_expert_block_hook(param, kind: str, idx: int, hook_fn: Callable):
        """
        给 expert block 的整体参数注册一个 hook，记录所有有梯度专家的最大值和均值。

        Args:
            param: nn.Parameter, shape [num_expert, ...]
            kind: "weight" or "bias"
            idx: 当前 LayerNorm 层的 idx
            hook_fn: Callable(name: str, value: Tensor) → Tensor
        """

        def hook(grad):
            grad_flat = grad.view(grad.size(0), -1)  # [num_expert, D]
            # 只保留有梯度的行（非零行）
            mask = grad_flat.abs().sum(dim=1) > 0  # [num_expert] boolean
            active_grads = grad_flat[mask]

            if active_grads.numel() > 0:
                norms = active_grads.norm(p=2, dim=1)  # [num_active_expert]
                max_norm = norms.max()
                mean_norm = norms.mean()
                hook_fn(f"layer{idx}/expert_{kind}_grad_max", max_norm)
                hook_fn(f"layer{idx}/expert_{kind}_grad_mean", mean_norm)

            return grad

        param.register_hook(hook)

    @staticmethod
    def register_router_separate_hooks(
        router: nn.Linear, name_prefix: str, hook_fn: Callable
    ):
        """
        为 router 的 weight 和 bias 分别注册梯度 hook，登记各自模长。

        Args:
            router: nn.Linear 实例
            name_prefix: 如 "layer0/router"
            hook_fn: Callable(name: str, norm_tensor: Tensor)
        """
        router.weight.register_hook(
            lambda grad: hook_fn(f"{name_prefix}_weight_grad_norm", grad.norm(2))
        )
        router.bias.register_hook(
            lambda grad: hook_fn(f"{name_prefix}_bias_grad_norm", grad.norm(2))
        )

    def get_trainable_params(self):
        params = [self.experts_bias, self.experts_weight]
        params.append(self.weight)
        params.append(self.bias)
        if hasattr(self,"router"):
            for name, p in self.router.named_parameters():
                params.append(p)
        return params

    def update_coeff(self, coeff: torch.Tensor):
        self.coeff = coeff if self.pass_through_coeff else coeff.detach()

    def get_topks(self, x: torch.Tensor):
        B = x.size(0)
        if isinstance(self.base_mod, nn.LayerNorm):
            # 对于ViT（Patch Embedding后），每个token向量直接flatten
            dims = tuple(i for i in range(1, x.ndim - 1))
            flat = x.mean(dim=dims)
        elif isinstance(self.base_mod, nn.BatchNorm2d) or isinstance(
            self.base_mod, nn.GroupNorm
        ):
            # 对于ResNet（Feature Map），每个channel spatial average
            flat = x.mean(dim=[2, 3])  # (B, C), spatial avg
        else:
            raise ValueError(f"Unknown architecture {self.arch}")
        if self.samplewise:
            prob = F.softmax(self.router(flat), dim=-1)
            biased = prob - self.penalty
            vals, topks = torch.topk(biased, self.topk, dim=-1)
            coeff = torch.zeros((B, self.num_expert), device=prob.device)
            if self.weight_by_prob:
                coeff.scatter_(1, topks, vals / vals.detach().sum(-1, keepdim=True))
            else:
                coeff.scatter_(1, topks, vals / vals.detach())
        else:
            prob = F.softmax(self.router(flat).mean(0), dim=-1)
            biased = prob - self.penalty
            vals, topks = torch.topk(biased, self.topk)
            coeff = torch.zeros(self.num_expert, device=prob.device)
            if self.weight_by_prob:
                coeff.scatter_(0, topks, vals / vals.detach().sum())
            else:
                coeff.scatter_(0, topks, vals / vals.detach())
            coeff = coeff.unsqueeze(0).repeat(B, 1)
            topks = topks.unsqueeze(0).repeat(B, 1)
        cnt = torch.bincount(topks.flatten(), minlength=self.num_expert)
        self.cnt += cnt
        self.penalty += cnt * self.route_penalty / (1 + self.decay * self.step)
        self.penalty -= self.penalty.min()
        self.lb_loss = self.num_expert * (prob.mean(0) * (cnt / cnt.sum())).sum()
        self.update_coeff(coeff)
        self.topks = topks
        self.route_prob_list.append(prob.detach().cpu())

    def step_once(self):
        self.step += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (self.router is not None) or (self.coeff is not None), (
            "需要路由器或融合系数"
        )
        if self.router is not None:
            self.get_topks(x)
        final_weight, final_bias = fuse_params(
            self.experts_weight,
            self.experts_bias,
            self.topks,
            self.coeff,
            self.weight,
            self.bias,
        )
        if self.logger is not None:
            self.logger(locals())

        if isinstance(self.base_mod, nn.LayerNorm):
            result = apply_layernorm(x, final_weight, final_bias, self.base_mod.eps)
            self.cls_feature = result[:, 0]
            self.shared_feature = F.layer_norm(
                x[:, 0],
                (x.shape[-1],),
                weight=self.weight,
                bias=self.bias,
                eps=self.base_mod.eps,
            )
            return result
        elif isinstance(self.base_mod, nn.BatchNorm2d):
            return apply_batchnorm(x, final_weight, final_bias, self.base_mod.eps)
        elif isinstance(self.base_mod, nn.GroupNorm):
            return apply_groupnorm(
                x, self.base_mod.num_groups, final_weight, final_bias, self.base_mod.eps
            )
        else:
            raise NotImplementedError(f"Unsupported norm module: {type(self.base_mod)}")


def switch_to_MoE(model, config: Config):
    idx = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            continue
        new_mod = MoENormalizationLayer(
            idx=idx,
            num_expert=config.algo.moetta.num_expert,
            shared_expert=config.algo.moetta.shared_expert,
            base_mod=mod,
            randomness=config.algo.moetta.randomness,
            self_router=True,
            samplewise=config.algo.moetta.samplewise,
            topk=config.algo.moetta.topk,
            weight_by_prob=config.algo.moetta.weight_by_prob,
            penalty=config.algo.moetta.route_penalty,
            decay=config.algo.moetta.decay,
            device=config.env.device,
            logger=None,
            grad_hook=config.algo.moetta.grad_hook,
            pass_through_coeff=config.algo.moetta.pass_through_coeff,
        )
        set_nested_attr(model, name, new_mod)
        idx += 1
