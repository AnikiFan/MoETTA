"""
Copyright to DeYO Authors, ICLR 2024 Spotlight (top-5% of the submissions)
built upon on Tent code.

Modified by Xiao Fan (xiaofan140@gmail.com) for the MoETTA project:
- extend to log statistics with wandb
- remove ImageNetMask related code
"""

from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import torchvision
import math
from einops import rearrange
import wandb

from .moe_normalization import MoENormalizationLayer
from config import Config


class DeYO(nn.Module):
    """DeYO online adapts a model by entropy minimization with entropy and PLPD filtering & reweighting during testing.
    Once DeYOed, a model adapts itself by updating on every forward.
    """

    def __init__(self, model, config: Config, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.steps = steps
        self.episodic = episodic

        self.deyo_margin = config.algo.deyo.margin_coeff * math.log(
            config.data.num_class
        )
        self.margin_e0 = config.algo.deyo.margin_e0_coeff * math.log(
            config.data.num_class
        )

        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model, self.optimizer
        )

    def forward(self, x, targets=None, flag=True, group=None):
        if self.episodic:
            self.reset()

        if targets is None:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward = forward_and_adapt_deyo(
                        x,
                        self.model,
                        self.config,
                        self.optimizer,
                        self.deyo_margin,
                        self.margin_e0,
                        targets,
                        flag,
                        group,
                    )
                else:
                    outputs = forward_and_adapt_deyo(
                        x,
                        self.model,
                        self.config,
                        self.optimizer,
                        self.deyo_margin,
                        self.margin_e0,
                        targets,
                        flag,
                        group,
                    )
        else:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward, corr_pl_1, corr_pl_2 = (
                        forward_and_adapt_deyo(
                            x,
                            self.model,
                            self.config,
                            self.optimizer,
                            self.deyo_margin,
                            self.margin_e0,
                            targets,
                            flag,
                            group,
                        )
                    )
                else:
                    outputs = forward_and_adapt_deyo(
                        x,
                        self.model,
                        self.config,
                        self.optimizer,
                        self.deyo_margin,
                        self.margin_e0,
                        targets,
                        flag,
                        group,
                    )
        if wandb.run is not None and wandb.run.summary is not None:
            backward_value = wandb.run.summary.get("backward", 0)
            final_backward_value = wandb.run.summary.get("final_backward", 0)

            wandb.run.summary["backward"] = backward_value + backward
            wandb.run.summary["final_backward"] = final_backward_value + final_backward
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )
        self.ema = None


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_deyo(
    x,
    model,
    config,
    optimizer,
    deyo_margin,
    margin,
    targets=None,
    flag=True,
    group=None,
):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    outputs = model(x)
    if not flag:
        return outputs.detach()

    optimizer.zero_grad()
    entropys = softmax_entropy(outputs)
    if config.algo.deyo.filter_ent:
        filter_ids_1 = torch.where((entropys < deyo_margin))
    else:
        filter_ids_1 = torch.where((entropys > -1))
    entropys = entropys[filter_ids_1]
    backward = len(entropys)
    if backward == 0:
        if targets is not None:
            return outputs.detach(), 0, 0, 0, 0
        return outputs.detach(), 0, 0

    x_prime = x[filter_ids_1]
    x_prime = x_prime.detach()
    if config.algo.deyo.aug_type == "occ":
        first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
        final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
        occlusion_window = final_mean.expand(
            -1, -1, config.algo.deyo.occulusion_size, config.algo.deyo.occulusion_size
        )
        x_prime[
            :,
            :,
            config.algo.deyo.row_start : config.algo.deyo.row_start
            + config.algo.deyo.occulusion_size,
            config.algo.deyo.column_start : config.algo.deyo.column_start
            + config.algo.deyo.occulusion_size,
        ] = occlusion_window
    elif config.algo.deyo.aug_type == "patch":
        resize_t = torchvision.transforms.Resize(
            (
                (x.shape[-1] // config.algo.deyo.patch_len)
                * config.algo.deyo.patch_len,
                (x.shape[-1] // config.algo.deyo.patch_len)
                * config.algo.deyo.patch_len,
            )
        )
        resize_o = torchvision.transforms.Resize((x.shape[-1], x.shape[-1]))
        x_prime = resize_t(x_prime)
        x_prime = rearrange(
            x_prime,
            "b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w",
            ps1=config.algo.deyo.patch_len,
            ps2=config.algo.deyo.patch_len,
        )
        perm_idx = torch.argsort(torch.rand(x_prime.shape[0], x_prime.shape[1]), dim=-1)
        x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1), perm_idx]
        x_prime = rearrange(
            x_prime,
            "b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)",
            ps1=config.algo.deyo.patch_len,
            ps2=config.algo.deyo.patch_len,
        )
        x_prime = resize_o(x_prime)
    elif config.algo.deyo.aug_type == "pixel":
        x_prime = rearrange(x_prime, "b c h w -> b c (h w)")
        x_prime = x_prime[:, :, torch.randperm(x_prime.shape[-1])]
        x_prime = rearrange(
            x_prime, "b c (ps1 ps2) -> b c ps1 ps2", ps1=x.shape[-1], ps2=x.shape[-1]
        )
    with torch.no_grad():
        outputs_prime = model(x_prime)

    prob_outputs = outputs[filter_ids_1].softmax(1)
    prob_outputs_prime = outputs_prime.softmax(1)

    cls1 = prob_outputs.argmax(dim=1)

    plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1, 1)) - torch.gather(
        prob_outputs_prime, dim=1, index=cls1.reshape(-1, 1)
    )
    plpd = plpd.reshape(-1)

    if config.algo.deyo.filter_plpd:
        filter_ids_2 = torch.where(plpd > config.algo.deyo.plpd_threshold)
    else:
        filter_ids_2 = torch.where(plpd >= -2.0)
    entropys = entropys[filter_ids_2]
    final_backward = len(entropys)

    if targets is not None:
        corr_pl_1 = (targets[filter_ids_1] == prob_outputs.argmax(dim=1)).sum().item()

    if final_backward == 0:
        del x_prime
        del plpd

        if targets is not None:
            return outputs.detach(), backward, 0, corr_pl_1, 0
        return outputs.detach(), backward, 0

    plpd = plpd[filter_ids_2]

    if targets is not None:
        corr_pl_2 = (
            (
                targets[filter_ids_1][filter_ids_2]
                == prob_outputs[filter_ids_2].argmax(dim=1)
            )
            .sum()
            .item()
        )

    if config.algo.deyo.reweight_ent or config.algo.deyo.reweight_plpd:
        coeff = config.algo.deyo.reweight_ent * (
            1 / (torch.exp(((entropys.clone().detach()) - margin)))
        ) + config.algo.deyo.reweight_plpd * (
            1 / (torch.exp(-1.0 * plpd.clone().detach()))
        )
        entropys = entropys.mul(coeff)
    loss = entropys.mean(0)

    if final_backward != 0:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()

    del x_prime
    del plpd

    if targets is not None:
        return outputs.detach(), backward, final_backward, corr_pl_1, corr_pl_2
    return outputs.detach(), backward, final_backward


def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        # if 'layer4' in nm:
        #     continue
        # if 'blocks.9' in nm:
        #     continue
        # if 'blocks.10' in nm:
        #     continue
        # if 'blocks.11' in nm:
        #     continue
        # if 'norm.' in nm:
        #     continue
        # if nm in ['norm']:
        #     continue

        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ["weight", "bias"]:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
        if isinstance(m, MoENormalizationLayer):
            params.extend(m.get_trainable_params())

    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with DeYO."""
    # train mode, because DeYO optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what DeYO updates
    model.requires_grad_(False)
    # configure norm for DeYO updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model
