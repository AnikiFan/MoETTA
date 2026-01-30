"""
Copyright to BECoTTA Authors.
Based on https://github.com/daeunni/becotta

Modified by Xiao Fan (xiaofan140@gmail.com) for the MoETTA project:
- tailor to work with MoETTA codebase
"""

from timm.models.vision_transformer import Block, VisionTransformer
from timm.models.swin_transformer import SwinTransformer
from timm.models.convnext import ConvNeXt
import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import wandb
import numpy as np
import math

from config import Config


class SimpleAdapter(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # Downconv
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)  # UPconv
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MoEBlock(nn.Module):
    def __init__(
        self,
        block: Block,
        idx: int,
        expert_num: int,
        MoE_hidden_dim: int,
        num_k: int,
        domain_num: int,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        if hasattr(block, "norm1"):
            dim = block.norm1.normalized_shape[0]
            device = block.norm1.weight.device
        elif hasattr(block, "norm"):
            dim = block.norm.normalized_shape[0]
            device = block.norm.weight.device
        else:
            raise ValueError("Cannot find normalization layer in the block.")
        self.idx = idx
        self.expert_num = expert_num
        self.MoE_hidden_dim = MoE_hidden_dim
        self.num_K = num_k
        self.domain_num = domain_num
        self.block = block
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.f_gate = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim, 2 * expert_num, bias=False)).to(device)
                for i in range(self.domain_num)
            ]
        )
        self.f_gate.apply(self._init_weights)
        expert_lists = []
        for _ in range(expert_num):
            tmp_adapter = SimpleAdapter(
                in_features=dim,
                hidden_features=MoE_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            ).to(device)
            tmp_adapter.apply(self._init_weights)
            expert_lists.append(tmp_adapter)

        self.adapter_experts = nn.ModuleList(expert_lists)
        self.register_buffer(
            "expert_sample_counter",
            torch.zeros(expert_num, dtype=torch.long).to(device),
        )

    def minmax_scaling(self, top_k_logits):
        """Per-sample min-max scaling over the k selected logits.
        top_k_logits: [B, K]
        """
        m1 = top_k_logits.amin(dim=1, keepdim=True)
        m2 = top_k_logits.amax(dim=1, keepdim=True)
        return (top_k_logits - m1) / (m2 - m1 + 1e-6)

    def forward(self, x):
        """
        x: [B, N, C]   (ViT token sequence format)
        Returns: same shape as input.
        Batch-size agnostic implementation.
        """
        # 1) Base block forward
        x = self.block(x)  # [B, N, C]
        permuted = False
        if len(x.shape) == 4 and x.shape[-1] == x.shape[-2]:
            permuted = True
            x = x.permute(0, 2, 3, 1)
        B = x.shape[0]

        # 2) Choose (or infer) which domain-gate to use per sample
        #    Here we keep your original random choice logic but make it batch-wise.
        #    Replace this with your own domain assignment if you have it.
        task_ids = torch.randint(
            low=0, high=self.domain_num, size=(B,), device=x.device
        )

        # 3) Compute gating logits for each sample using its selected domain gate
        #    self.f_gate[d](x[b]) expects [N, C] -> [N, 2*E]
        total_w_list = []
        for b in range(B):
            total_w_list.append(self.f_gate[task_ids[b]](x[b]))  # [N, 2E]
        total_w = torch.stack(total_w_list, dim=0).flatten(1, -2)  # [B, N, 2E]

        clean_logits, raw_noise_stddev = total_w.chunk(2, dim=-1)  # each [B, N, E]
        noise_stddev = F.softplus(raw_noise_stddev) + 1e-2
        eps = torch.randn_like(clean_logits)
        logits = clean_logits + eps * noise_stddev  # [B, N, E]

        # 4) Aggregate token-level logits to expert-level scores per sample
        exp_wise_sum = logits.sum(dim=1)  # [B, E]

        # 5) Top-K expert selection per sample
        K = min(self.num_K, self.expert_num)
        top_k_logits, top_k_indices = exp_wise_sum.topk(K, dim=1)  # both [B, K]
        with torch.no_grad():
            flat_indices = top_k_indices.flatten()  # [B * K]
            unique_ids, counts = flat_indices.unique(return_counts=True)
            # self.expert_sample_counter.index_add_(0, unique_ids, counts)
        # 6) Normalize gates (softmax after optional min-max scaling)
        top_k_logits_scaled = self.minmax_scaling(top_k_logits)
        top_k_gates = self.softmax(top_k_logits_scaled)  # [B, K]

        # 7) Build a dense gate matrix [B, E] with zeros for non-selected experts
        gates_full = x.new_zeros(B, self.expert_num)
        gates_full.scatter_(1, top_k_indices, top_k_gates)
        # gates_full: [B, E]

        # 8) Forward all experts once (vectorized)
        # adapter_experts[i](x) -> [B, N, C]
        expert_outs = torch.stack(
            [adapter(x) for adapter in self.adapter_experts], dim=1
        )  # [B, E, N, C]

        # 9) Weighted sum over experts
        while gates_full.dim() < expert_outs.dim():
            gates_full = gates_full.unsqueeze(-1)  # [B, E, 1, 1]
        tot_x = (gates_full * expert_outs).sum(dim=1)  # [B, N, C]
        if wandb.run is not None and wandb.run.summary is not None:
            wandb.run.summary[f"layer{self.idx}_expert_cnt"] = (
                self.expert_sample_counter.cpu()
            )

        if permuted:
            tot_x = tot_x.permute(0, 3, 1, 2)
            x = x.permute(0, 3, 1, 2)

        return x + tot_x

    # def minmax_scaling(self, top_k_logits) :
    #     m1 = top_k_logits.min()
    #     m2 = top_k_logits.max()
    #     return (top_k_logits-m1)/ (m2-m1)

    def one_hot_encoding(self, index, num_classes):
        one_hot = np.zeros(num_classes)
        one_hot[index] = 1
        return one_hot

    # def forward(self,x):
    #     tot_x = torch.zeros_like(x)
    #     x = self.block(x)
    #     task_bh =  torch.randint(low=0, high=self.expert_num, size=(1,)).item()
    #     total_w = self.f_gate[task_bh](x)
    #     clean_logits, raw_noise_stddev = total_w.chunk(2, dim=-1)
    #     noise_stddev = F.softplus(raw_noise_stddev) + 1e-2
    #     eps = torch.randn_like(clean_logits)
    #     logits = clean_logits + eps * noise_stddev
    #     exp_wise_sum = logits.sum(dim=1)

    #     # select topk
    #     top_logits, top_indices = exp_wise_sum.topk(min(self.num_K + 1, self.expert_num), dim=1)
    #     top_k_logits = top_logits[:, :self.num_K]      # [batch, k]
    #     top_k_indices = top_indices[:, :self.num_K]    # [batch, k] -> selected experts

    #     if len(top_k_logits) > 1 :
    #         top_k_gates = self.softmax(self.minmax_scaling(top_k_logits))       # [batch, k] -> probabilities
    #     else :
    #         top_k_gates = self.softmax(top_k_logits)
    #     assert top_k_indices.shape == top_k_gates.shape

    #     # adapter weighted output
    #     for idx in range(len(top_k_indices[0])) :
    #         tot_x += top_k_gates[0][idx].item() * self.adapter_experts[idx](x)

    #     x = x + tot_x
    #     return x

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class BECoTTA(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, thr_coeff=0.4):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0
        self.episodic = episodic
        self.thr_coeff = thr_coeff
        self.forward_cnt = 0

    def forward(self, x, y=None):
        if self.steps > 0:
            for _ in range(self.steps):
                outputs, cnt = forward_and_adapt(
                    x, self.model, self.optimizer, self.thr_coeff
                )
            self.forward_cnt += cnt
            if wandb.run is not None and wandb.run.summary is not None:
                wandb.run.summary["forward_cnt"] = self.forward_cnt
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
        return outputs


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x / temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, thr_coeff):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    # adapt
    entropys = softmax_entropy(outputs)
    idx = entropys < thr_coeff * math.log(1000)
    forward_cnt = idx.sum().item()
    loss = entropys[idx].mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs, forward_cnt


def collect_params(model):
    params = []
    names = []
    if isinstance(model, VisionTransformer):
        for block in model.blocks:
            for name, param in block.f_gate.named_parameters():
                names.append(name)
                params.append(param)
            for name, param in block.adapter_experts.named_parameters():
                names.append(name)
                params.append(param)
    elif isinstance(model, SwinTransformer):
        for layer in model.layers:
            for block in layer.blocks:
                for name, param in block.f_gate.named_parameters():
                    names.append(name)
                    params.append(param)
                for name, param in block.adapter_experts.named_parameters():
                    names.append(name)
                    params.append(param)
    elif isinstance(model, ConvNeXt):
        for stage in model.stages:
            for block in stage.blocks:
                for name, param in block.f_gate.named_parameters():
                    names.append(name)
                    params.append(param)
                for name, param in block.adapter_experts.named_parameters():
                    names.append(name)
                    params.append(param)
    else:
        raise NotImplementedError
    return params, names


def configure_model(model, config: Config):
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    if isinstance(model, VisionTransformer):
        blocks = []
        for i, block in enumerate(model.blocks):
            block = MoEBlock(
                block,
                i,
                config.algo.becotta.expert_num,
                config.algo.becotta.MoE_hidden_dim,
                config.algo.becotta.num_k,
                config.algo.becotta.domain_num,
            )
            block.f_gate.requires_grad_(True)
            block.adapter_experts.requires_grad_(True)
            blocks.append(block)
        model.blocks = nn.Sequential(*blocks)
    if isinstance(model, SwinTransformer):
        for layer in model.layers:
            blocks = []
            for i, block in enumerate(layer.blocks):
                block = MoEBlock(
                    block,
                    i,
                    config.algo.becotta.expert_num,
                    config.algo.becotta.MoE_hidden_dim,
                    config.algo.becotta.num_k,
                    config.algo.becotta.domain_num,
                )
                block.f_gate.requires_grad_(True)
                block.adapter_experts.requires_grad_(True)
                blocks.append(block)
            layer.blocks = nn.Sequential(*blocks)
    if isinstance(model, ConvNeXt):
        for stage in model.stages:
            blocks = []
            for i, block in enumerate(stage.blocks):
                block = MoEBlock(
                    block,
                    i,
                    config.algo.becotta.expert_num,
                    config.algo.becotta.MoE_hidden_dim,
                    config.algo.becotta.num_k,
                    config.algo.becotta.domain_num,
                )
                block.f_gate.requires_grad_(True)
                block.adapter_experts.requires_grad_(True)
                blocks.append(block)
            stage.blocks = nn.Sequential(*blocks)
    return model
