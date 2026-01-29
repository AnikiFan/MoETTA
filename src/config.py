from dataclasses import dataclass
from typing import Literal
from pathlib import Path


@dataclass
class EnvironmentConfig:
    project: str = "MoETTA"
    group: str = ""
    name: str = ""
    num_cpus: float = 40.0
    num_gpus: float = 1.0
    device: str = "cuda"
    tags: tuple[str] = ()
    local: bool = False
    notes: str = ""
    job_type: str = "train"
    wandb_mode: Literal["online", "offline", "disabled", "shared"] = "online"
    data: Path = Path("~/workspace/MoETTA/data/imagenet-1k_2012")
    data_sketch: Path = Path("~/workspace/MoETTA/data/imagenet-sketch/sketch")
    data_adv: Path = Path("~/workspace/MoETTA/data/imagenet-a")
    data_v2: Path = Path("~/workspace/MoETTA/data/imagenetv2")
    data_corruption: Path = Path("~/workspace/MoETTA/data/imagenet-c")
    data_rendition: Path = Path("~/workspace/MoETTA/data/imagenet-r")
    data_cifar100_c: Path = Path("~/workspace/MoETTA/data/cifar100-c")
    data_cifar10_c: Path = Path("~/workspace/MoETTA/data/cifar10-c")


@dataclass
class TrainingConfig:
    optimizer: Literal["SGD", "AdamW"] = "SGD"
    batch_size: int = 64
    seed: int = 42
    workers: int = 8
    ray_tune_config: Path = ""


@dataclass
class ModelConfig:
    model: Literal[
        "vit_base_patch16_224",
        "vit_tiny_patch16_224",
        "vit_small_patch16_224",
        "vit_large_patch16_224",
        "resnet18",
        "resnet50",
        "resnet50_gn",
        "swin_base_patch4_window7_224",
        "convnext_base",
    ] = "vit_base_patch16_224"


@dataclass
class DataConfig:
    num_class: int = 1000
    used_data_num: int = -1  # -1 means to use all the data
    shuffle: bool = True
    level: Literal[1, 2, 3, 4, 5] = 5  # corruption level for corrupted dataset
    corruption: Literal[
        "rendition",
        "sketch",
        "imagenet_a",
        "imagenet_c_val_mix",
        "original",
        "imagenet_c_test_mix",
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
        "speckle_noise",
        "spatter",
        "gaussian_blur",
        "saturate",
        "potpourris",
        "potpourris+",
        "cifar10-c",
        "cifar100-c",
    ] = "imagenet_c_test_mix"


@dataclass
class BECoTTAConfig:
    lr: float = 0.001
    expert_num: int = 6
    MoE_hidden_dim: int = 2
    num_k: int = 6
    domain_num: int = 1


@dataclass
class MGTTAConfig:
    mgg_path: Path = Path("~/workspace/MoETTA/artifacts/mgg_ckpt.pth")
    ttt_hidden_size: int = 8
    num_attention_heads: int = 1
    norm_dim: int = 768
    train_info_path: Path = Path("~/workspace/MoETTA/artifacts/train_info.pt")
    lr: float = 0.001


@dataclass
class CoTTAConfig:
    lr: float = 0.001


@dataclass
class SARConfig:
    lr: float = 0.001
    margin_e0_coeff: float = 0.4
    reset_constant_em: float = 0.005


@dataclass
class DeYOConfig:
    lr: float = 0.001
    margin_coeff: float = 0.4
    margin_e0_coeff: float = 0.5
    filter_ent: bool = True
    filter_plpd: bool = True
    reweight_ent: bool = True
    reweight_plpd: bool = True
    aug_type: Literal["occ", "patch", "pixel"] = "patch"
    patch_len: int = 4
    occulusion_size: int = 112
    row_start: int = 56
    column_start: int = 56
    plpd_threshold: float = 0.2


@dataclass
class EATAConfig:
    lr: float = 0.001
    fisher_size: int = 2000
    fisher_alpha: float = 2000.0
    e_margin_coeff: float = 0.4
    d_margin: float = 0.05


@dataclass
class TentConfig:
    lr: float = 0.001


@dataclass
class MoETTAConfig:
    lr: float = 0.001

    randomness: float = 0.0
    """Ratio of expert random initialization norm to pretrained parameter norm."""

    num_expert: int = 10
    """Number of experts."""

    topk: int = 5
    """Number of activated experts."""

    route_penalty: float = 0.0
    """Constant used in DeepseekV3 loss-free routing balancing method."""

    weight_by_prob: bool = False
    """Whether to use normalized router softmax values as coefficients during expert fusion. If False, use uniform coefficients."""

    shared_expert: bool = False
    """Whether to train shared experts, i.e., whether to train pretrained parameters."""

    ethr_coeff: float = 1.0
    """Coefficient used for entropy filtering threshold."""

    lb_coeff: float = 1.0
    """Coefficient before load balancing loss."""

    decay: float = 0.0
    """Decay coefficient for route_penalty."""

    self_router: bool = True
    """Whether to have a router for each LN."""

    weight_by_entropy: bool = False
    """Whether to weight by entropy."""

    e_margin_coeff: float = 0.4
    """Bias term coefficient in entropy weighting."""

    clip_grad: bool = False
    """Whether to clip gradients."""

    grad_hook: bool = False
    """Whether to log gradient norm information to wandb."""

    expert_grad_clip: float = 1.5
    """Threshold for clipping expert gradients."""

    router_grad_clip: float = 1.5
    """Threshold for clipping router gradients."""

    dynamic_threshold: bool = True
    """Whether to use dynamic thresholding."""

    samplewise: bool = True
    """Whether to route on a per-sample basis."""

    log_matrix_step: int = 100
    """Step interval for logging matrices."""

    disabled_layer: str = ""

    normal_layer: str = ""

    pass_through_coeff: bool = True

    log_detail: bool = False

    early_stop: bool = False

    dynamic_lb: bool = True

    global_router_idx: float = -1.0

    def __post_init__(self):
        self.disabled_layer = (
            list(range(int(self.disabled_layer.split('-')[0]), int(self.disabled_layer.split('-')[1]) + 1))
            if '-' in self.disabled_layer
            else [int(x) for x in self.disabled_layer.split(',') if x]
        )
        self.normal_layer = (
            list(range(int(self.normal_layer.split('-')[0]), int(self.normal_layer.split('-')[1]) + 1))
            if '-' in self.normal_layer
            else [int(x) for x in self.normal_layer.split(',') if x]
        )

@dataclass
class AlgorithmConfig:
    moetta: MoETTAConfig
    eata: EATAConfig
    tent: TentConfig
    deyo: DeYOConfig
    sar: SARConfig
    cotta: CoTTAConfig
    becotta: BECoTTAConfig
    mgtta: MGTTAConfig
    algorithm: Literal[
        "tent", "eata", "deyo", "sar", "cotta", "mgtta", "becotta", "moetta", "noadapt"
    ] = "tent"
    switch_to_MoE: bool = False


@dataclass
class TuneConfig:
    search_space: Path = ""
    gpu_per_trial: float = 1.0
    cpu_per_trial: float = 4.0
    num_samples: int = 10
    max_t: int = 3600
    job_name: str = ""
    scheduler: Literal["HyperBandScheduler"] = "HyperBandScheduler"
    search_algorithm: Literal["ax", "bayes", "optuna", "basic"] = "optuna"
    mode: Literal["min", "max"] = "max"
    metric: str = "overall_accuracy"


@dataclass
class Config:
    env: EnvironmentConfig
    train: TrainingConfig
    model: ModelConfig
    data: DataConfig
    algo: AlgorithmConfig
    tune: TuneConfig
