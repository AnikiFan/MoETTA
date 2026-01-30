import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import List, Callable
import wandb

from ..utils import set_nested_attr, softmax_entropy
from .moe_normalization import MoENormalizationLayer
from .router import MLPRouter
from config import Config


class MoETTA(nn.Module):
    """
    以batch为单位来决定分配的专家
    """

    def __init__(
        self,
        model,
        config: Config,
        num_expert: int,
        topk: int,
        dynamic_threshold: bool = True,
        dynamic_lb: bool = True,
        lb_coeff: float = 1.0,
        weight_by_prob: bool = True,
        weight_by_entropy: bool = True,
        randomness: float = 0,
        shared_expert: bool = True,
        route_penalty: float = 0.0,
        decay: float = 0.0,
        self_router: bool = False,
        samplewise: bool = False,
        moe_logger: Callable = None,
        grad_hook: bool = False,
        disabled_layer: List[int] = None,
        normal_layer: List[int] = None,
        pass_through_coeff: bool = True,
        global_router_idx: int = -1,
        device: str = "cuda",
    ):
        """
        初始化

        Args:
            model: 用于TTA的Vit
            num_expert: 专家数
            topk: 路由的专家数
            weight_by_prob: 是否将路由概率作为融合时的系数. Defaults to True.
            randomness: 随机初始化专家时模长和预训练参数模长之比. Defaults to 0.
            shared_expert: 是否有共享专家（这里把预训练参数作为共享专家）. Defaults to True.
            route_penalty: auxiliary loss free所用超参数. Defaults to 0.0.
            device: 所用设备. Defaults to "cuda".
        """
        super().__init__()
        assert num_expert >= topk, "num_expert must greater or equal than topk"
        assert route_penalty >= 0, "route penalty should be positive"
        self.device = device
        self.config = config
        self.model = model.to(self.device)
        self.num_expert = num_expert
        if self_router and global_router_idx < 0:
            self.router = None
        else:
            self.router = MLPRouter(
                num_expert, self.model.patch_embed.proj.weight.shape[0], device
            )
            self.cnt = torch.zeros(self.num_expert).to(device)
        self.topk = topk
        self.weight_by_prob = weight_by_prob
        self.randomness = randomness
        self.shared_expert = shared_expert
        self.dynamic_threshold = dynamic_threshold
        self.dynamic_lb = dynamic_lb
        self.weight_by_entropy = weight_by_entropy
        self.lb_coeff = lb_coeff
        self.route_penalty = route_penalty
        self.decay = decay
        self.step = 0
        self.attention_blocks = getattr(self.model, "blocks", None)
        self.self_router = self_router
        self.samplewise = samplewise
        self.penalty = torch.zeros(self.num_expert).to(self.device)
        self.moe_logger = moe_logger
        self.grad_hook = grad_hook
        self.disabled_layer = disabled_layer
        self.normal_layer = normal_layer
        self.pass_through_coeff = pass_through_coeff
        self.criterion_mse = nn.MSELoss(reduction="none").cuda()
        self.global_router_idx = global_router_idx
        self.normal_layers = []
        self.output_list = []
        self.entropys = []
        self.probs = []
        self.biased_probs = []
        self.cnt = torch.zeros(self.num_expert).to(self.device)
        self.filtered_cnt = 0
        self.construct_model()
        self.optimizer = torch.optim.SGD(
            self.get_params(), lr=config.algo.moetta.lr, momentum=0.9
        )

    def patch_embedding(self, x):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        # x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        return x

    def classifier(self, x):
        x = self.model.norm(x)
        x = self.model.pool(x)
        x = self.model.fc_norm(x)
        # x = self.model.head_drop(x)
        x = self.model.head(x)
        return x

    def construct_model(self):
        """
        Replace Normalization with MoENormalization
        """

        def router_hook(module, input):
            coeff, topks, biased_prob, route_prob = self.get_coeff_topks(input[0])
            self.biased_prob = biased_prob
            self.route_prob = route_prob
            self.set_coeff(coeff)
            self.set_topks(topks)

        self.model.requires_grad_(False)
        idx = 0

        for name, mod in self.model.named_modules():
            if not isinstance(mod, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                continue
            if idx in self.disabled_layer and idx in self.normal_layer:
                raise ValueError(f"{idx} is both in disabled layer and normal layer")
            if idx in self.disabled_layer or idx < self.global_router_idx:
                idx += 1
                continue
            if idx in self.normal_layer:
                idx += 1
                mod.requires_grad_(True)
                self.normal_layers.append(mod)
                continue
            new_mod = MoENormalizationLayer(
                idx=idx,
                num_expert=self.num_expert,
                shared_expert=self.shared_expert,
                base_mod=mod,
                randomness=self.randomness,
                self_router=self.self_router,
                samplewise=self.samplewise,
                topk=self.topk,
                weight_by_prob=self.weight_by_prob,
                penalty=self.route_penalty,
                decay=self.decay,
                device=self.device,
                logger=self.moe_logger,
                grad_hook=self.grad_hook,
                pass_through_coeff=self.pass_through_coeff,
            )
            if idx == self.global_router_idx:
                new_mod.register_forward_pre_hook(router_hook)
            set_nested_attr(self.model, name, new_mod)
            idx += 1
        self.moe_layer = [
            mod
            for mod in self.model.modules()
            if isinstance(mod, MoENormalizationLayer)
        ]

    def get_coeff_topks(self, embedding):
        B = embedding.size(0)
        if self.samplewise:
            route_prob = F.softmax(self.router(torch.mean(embedding, dim=1)), dim=-1)
            biased_prob = route_prob - self.penalty
            prob, topks = torch.topk(biased_prob, self.topk, dim=-1)
            cnt = torch.bincount(topks.flatten(), minlength=self.num_expert)
            self.cnt += cnt
            self.penalty += cnt * self.route_penalty / (1 + self.decay * self.step)
            self.penalty -= self.penalty.min()
            coeff = torch.zeros(
                (embedding.shape[0], self.num_expert), device=self.device
            )
            coeff = coeff.scatter(
                1,
                topks,
                prob / prob.detach().sum(-1, keepdim=True)
                if self.weight_by_prob
                else prob / prob.detach(),
            )
            importance = route_prob.mean(dim=0)
            load = cnt / cnt.sum()
            self.lb_loss = self.num_expert * torch.sum(importance * load)
        else:
            route_prob = F.softmax(
                self.router(torch.mean(embedding, dim=[0, 1])), dim=-1
            )
            biased_prob = route_prob - self.penalty
            prob, topks = torch.topk(biased_prob, self.topk)
            self.penalty[topks] += self.route_penalty / (1 + self.decay * self.step)
            self.cnt[topks] += 1
            self.penalty -= self.penalty.min()
            coeff = torch.zeros(self.num_expert, device=self.device)
            coeff = coeff.scatter(
                0,
                topks,
                prob / prob.detach().sum()
                if self.weight_by_prob
                else prob / prob.detach(),
            )
            coeff = coeff.unsqueeze(0).repeat(B, 1)
            topks = topks.repeat(B, 1)
        return coeff, topks, biased_prob, route_prob

    @torch.enable_grad()
    def forward(self, x):
        output = self.model(x)
        lb_loss = self.collect_lb_loss().sum()
        wandb.log({"aux_loss": lb_loss}, step=self.step)
        entropy = softmax_entropy(output)
        self.output_list.append(output.detach().cpu())
        mean_entropy = entropy.detach().mean()
        min_entropy = entropy.detach().min().item()
        if len(self.entropys) == 0:
            self.entropys.append(mean_entropy)
        else:
            old_avg = torch.stack(self.entropys).detach().mean().item()
            self.entropys.append(mean_entropy)
            new_avg = torch.stack(self.entropys).detach().mean().item()
        if self.dynamic_threshold:
            if len(self.entropys) == 1:
                self.threshold = mean_entropy
            else:
                self.threshold *= new_avg / old_avg
        if self.dynamic_lb:
            if len(self.entropys) == 1:
                self.lb_coeff = mean_entropy * self.lb_coeff
            else:
                self.lb_coeff *= new_avg / old_avg
        wandb.log({"threshold": self.threshold}, step=self.step)
        wandb.log({"lb_coeff": self.lb_coeff}, step=self.step)
        entropy = entropy[entropy <= self.threshold]
        coeff = (
            1 / (torch.exp(entropy.detach() - self.e_margin_coeff * math.log(1000)))
            if self.weight_by_entropy
            else 1
        )
        wandb.log({"entropy": mean_entropy, "min_entropy": min_entropy}, step=self.step)
        # DeYO set 0.5*log(1000)
        # SAR set 0.4*log(1000)
        self.optimizer.zero_grad()
        ((coeff * entropy).mean() + self.lb_coeff * lb_loss).backward()
        self.optimizer.step()
        self.filtered_cnt += self.config.train.batch_size - entropy.numel()
        wandb.log(
            {
                "overall_filter_ratio": self.filtered_cnt
                / (self.config.train.batch_size * (self.step + 1)),
                "batch_filter_ratio": (self.config.train.batch_size - entropy.numel())
                / self.config.train.batch_size,
            },
            step=self.step,
        )
        self.step_once()

        if not self.self_router:
            self.probs.append(self.model.route_prob)
            self.biased_probs.append(self.model.biased_prob)
            route_prob_max, selected = torch.topk(
                self.model.biased_prob, self.config.train.topk
            )
            self.cnt[selected] += 1
            table = wandb.Table(
                data=[[label, val] for label, val in enumerate(self.cnt)],
                columns=["expert", "cnt"],
            )
            wandb.log(
                {
                    "expert_distribution": wandb.plot.bar(
                        table, "label", "value", title="Expert Distribution"
                    )
                },
                step=self.step,
            )
        return output

    def set_coeff(self, coeff: torch.Tensor):
        """
        set current expert
        """
        for layer in self.moe_layer:
            layer.update_coeff(coeff)

    def set_topks(self, topks: torch.Tensor):
        """
        set topks
        """
        for layer in self.moe_layer:
            layer.topks = topks

    def get_expert(self, experts) -> List[nn.Parameter]:
        """
        get current expert
        """
        params = []
        for mod in self.moe_layer:
            for exp in experts:
                params += [
                    mod.experts_weight[exp],
                    mod.experts_bias[exp],
                ]
        return params

    def get_shared_expert(self) -> List[nn.Parameter]:
        """
        get shared expert, i.e., the pretrained parameter of LN
        """
        params = []
        for mod in self.moe_layer:
            params.extend([mod.weight, mod.bias])
        return params

    def collect_lb_loss(self):
        if not self.self_router:
            return self.lb_loss
        loss = []
        for mod in self.moe_layer:
            loss.append(mod.lb_loss)
        return torch.stack(loss)

    def step_once(self):
        for mod in self.moe_layer:
            mod.step_once()

    def get_params(self):
        params = []
        if self.router is not None:
            params.extend(self.router.get_params())
        for mod in self.model.modules():
            if isinstance(mod, MoENormalizationLayer):
                params.extend(mod.get_trainable_params())
        for m in self.normal_layers:
            for name, p in m.named_parameters():
                if name in ["weight", "bias"]:  # weight is scale, bias is shift
                    params.append(p)
        return params
