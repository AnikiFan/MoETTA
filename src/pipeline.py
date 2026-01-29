import torch
import timm
import math
from tqdm import tqdm
import wandb
import ray.tune as tune
from torch.utils.data import Subset
import torch.nn as nn

from .config import Config
from .utils import (
    wandb_log,
    deterministic,
    show_config,
    timer,
    mem_trace,
    CumulativeTimer,
    count_correct,
    get_logger,
)
from .adaptation import tent
from .adaptation import eata
from .adaptation import sar
from .adaptation import deyo
from .adaptation import cotta
from .adaptation import mgtta
from .adaptation import becotta
from .adaptation.sam import SAM
from .dataset.dataset import get_data, prepare_test_data
from .adaptation.vpt import FOAViT
from .adaptation.moetta import MoETTA
from .adaptation.moe_normalization import switch_to_MoE


@CumulativeTimer
def validate(val_loader, model, config: Config):
    n_top1, n_top5, n_sample = 0, 0, 0
    for i, batch in enumerate(tqdm(val_loader)):
        images, target = batch[0].cuda(), batch[1].cuda()

        if config.algo.algorithm != "mgtta":
            with torch.no_grad():
                output = model(images)
        else:
            output, loss = model(images)
        # measure accuracy and record loss
        acc1, acc5 = count_correct(output, target, topk=(1, 5))
        n_top1 += acc1
        n_top5 += acc5
        n_sample += images.shape[0]

        del output
        del target
        # measure elapsed time
        wandb.log(
            dict(
                batch_accuracy_1=acc1 / images.shape[0],
                batch_accuracy_5=acc5 / images.shape[0],
                overall_accuracy_1=n_top1 / n_sample,
                overall_accuracy_5=n_top5 / n_sample,
            ),
            step=i,
        )
        tune.report(
            dict(
                batch_accuracy_1=acc1 / images.shape[0],
                batch_accuracy_5=acc5 / images.shape[0],
                overall_accuracy_1=n_top1 / n_sample,
                overall_accuracy_5=n_top5 / n_sample,
                step=i,
            )
        )

    return n_top1 / n_sample


def configure_model(config: Config):
    net = timm.create_model(config.model.model, pretrained=True)
    net = net.cuda()
    net.eval()
    net.requires_grad_(False)

    match config.algo.algorithm:
        case "tent":
            net = tent.configure_model(net)
            if config.algo.switch_to_MoE:
                switch_to_MoE(net, config)
            params, _ = tent.collect_params(net)
            optimizer = torch.optim.SGD(params, config.algo.tent.lr, momentum=0.9)
            adapt_model = tent.Tent(net, optimizer)
        case "eata":
            # compute fisher informatrix
            fisher_dataset = get_data("original", config)
            fisher_dataset = Subset(
                fisher_dataset,
                torch.randperm(len(fisher_dataset))[: config.algo.eata.fisher_size],
            )
            fisher_loader = torch.utils.data.DataLoader(
                fisher_dataset,
                batch_size=config.train.batch_size,
                shuffle=config.data.shuffle,
                num_workers=config.train.workers,
                pin_memory=True,
            )
            fisher_loader = torch.utils.data.DataLoader(
                fisher_dataset,
                batch_size=config.train.batch_size,
                shuffle=config.data.shuffle,
                num_workers=config.train.workers,
                pin_memory=True,
            )

            net = eata.configure_model(net)
            if config.algo.switch_to_MoE:
                switch_to_MoE(net, config)
            params, param_names = eata.collect_params(net)
            ewc_optimizer = torch.optim.SGD(params, 0.001)
            fishers = {}
            train_loss_fn = nn.CrossEntropyLoss().cuda()
            for iter_, (images, targets) in enumerate(fisher_loader, start=1):
                if torch.cuda.is_available():
                    targets = targets.cuda(non_blocking=True)
                    images = images.cuda(non_blocking=True)
                outputs = net(images)
                _, targets = outputs.max(1)
                loss = train_loss_fn(outputs, targets)
                loss.backward()
                for name, param in net.named_parameters():
                    if param.grad is not None:
                        if iter_ > 1:
                            fisher = (
                                param.grad.data.clone().detach() ** 2 + fishers[name][0]
                            )
                        else:
                            fisher = param.grad.data.clone().detach() ** 2
                        if iter_ == len(fisher_loader):
                            fisher = fisher / iter_
                        fishers.update({name: [fisher, param.data.clone().detach()]})
                ewc_optimizer.zero_grad()
            del ewc_optimizer
            optimizer = torch.optim.SGD(params, config.algo.eata.lr, momentum=0.9)
            adapt_model = eata.EATA(
                net,
                optimizer,
                fishers,
                config.algo.eata.fisher_alpha,
                e_margin=config.algo.eata.e_margin_coeff
                * math.log(config.data.num_class),
                d_margin=config.algo.eata.d_margin,
            )
        case "deyo":
            net = deyo.configure_model(net)
            if config.algo.switch_to_MoE:
                switch_to_MoE(net, config)
            params, _ = deyo.collect_params(net)
            optimizer = torch.optim.SGD(params, config.algo.deyo.lr, momentum=0.9)
            adapt_model = deyo.DeYO(net, config, optimizer)
        case "sar":
            net = sar.configure_model(net)
            if config.algo.switch_to_MoE:
                switch_to_MoE(net, config)
            params, _ = sar.collect_params(net)
            base_optimizer = torch.optim.SGD
            optimizer = SAM(params, base_optimizer, lr=config.algo.sar.lr, momentum=0.9)
            # NOTE: set margin_e0 to 0.4*math.log(200) on ImageNet-R
            adapt_model = sar.SAR(
                net,
                optimizer,
                margin_e0=config.algo.sar.margin_e0_coeff
                * math.log(config.data.num_class),
                reset_constant_em=config.algo.sar.reset_constant_em,
            )
        case "cotta":
            net = cotta.configure_model(net)
            params, _ = cotta.collect_params(net)
            optimizer = torch.optim.SGD(params, lr=config.algo.cotta.lr, momentum=0.9)
            adapt_model = cotta.CoTTA(net, optimizer, steps=1, episodic=False)
        case "mgtta":
            net = FOAViT(net).cuda()
            mgg = mgtta.create_mgg(
                config.algo.mgtta.mgg_path,
                hidden_size=config.algo.mgtta.ttt_hidden_size,
                num_attention_heads=config.algo.mgtta.num_attention_heads,
            ).cuda()
            adapt_model = mgtta.MGTTA(
                net, mgg, config.algo.mgtta.lr, norm_dim=config.algo.mgtta.norm_dim
            )
            train_set = get_data("original", config)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=config.train.batch_size,
                shuffle=config.data.shuffle,
                num_workers=config.train.workers,
                pin_memory=True,
            )
            adapt_model.obtain_origin_stat(
                train_loader, config.algo.mgtta.train_info_path
            )
            adapt_model.configure_model()
        case "becotta":
            net = becotta.configure_model(net, config)
            params, _ = becotta.collect_params(net)
            optimizer = torch.optim.Adam(
                params, config.algo.becotta.lr, betas=(0.9, 0.999)
            )
            adapt_model = becotta.BECoTTA(net, optimizer)
        case "noadapt":
            adapt_model = net
        case "moetta":
            adapt_model = MoETTA(
                net,
                config,
                num_expert=config.algo.moetta.num_expert,
                topk=config.algo.moetta.topk,
                dynamic_threshold=config.algo.moetta.dynamic_threshold,
                dynamic_lb=config.algo.moetta.dynamic_lb,
                lb_coeff=config.algo.moetta.lb_coeff,
                weight_by_prob=config.algo.moetta.weight_by_prob,
                weight_by_entropy=config.algo.moetta.weight_by_entropy,
                randomness=config.algo.moetta.randomness,
                shared_expert=config.algo.moetta.shared_expert,
                route_penalty=config.algo.moetta.route_penalty,
                decay=config.algo.moetta.decay,
                self_router=config.algo.moetta.self_router,
                samplewise=config.algo.moetta.samplewise,
                moe_logger=get_logger(config),
                grad_hook=config.algo.moetta.grad_hook,
                disabled_layer=config.algo.moetta.disabled_layer,
                normal_layer=config.algo.moetta.normal_layer,
                pass_through_coeff=config.algo.moetta.pass_through_coeff,
                global_router_idx=config.algo.moetta.global_router_idx,
                device=config.env.device
            )
        case _:
            raise ValueError("Invalid algorithm!")

    return adapt_model


@wandb_log
@show_config
@deterministic
@mem_trace
@timer
def pipeline(config: Config):
    model = configure_model(config)
    val_dataset, val_loader = prepare_test_data(config)
    acc = validate(val_loader, model, config)
    return acc
