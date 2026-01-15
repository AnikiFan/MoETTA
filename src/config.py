from dataclasses import dataclass
from typing import Literal
from pathlib import Path

@dataclass
class EnvironmentConfig:
    project: str = ""
    group: str = ""
    name: str = ""
    num_cpus: float = 40.0
    num_gpus: float = 1.0
    data_path: Path = Path("/home/share/dataset/ILSVRC/imagenet-c")
    device: str = "cuda"
    tags: tuple[str] = ()
    local: bool = False
    notes: str = ""
    job_type: str = "train"
    wandb_mode: Literal["online","offline","disabled","shared"] = "online"
    data: Path = Path('./dataset/imagenet-1k_2012')
    data_sketch: Path = Path('./dataset/imagenet-sketch/sketch')
    data_adv: Path = Path('./dataset/imagenet-a')
    data_v2: Path = Path('./dataset/imagenetv2')
    data_corrupted: Path = Path('./dataset/imagenet-c')
    data_rendition: Path = Path('./dataset/imagenet-r')
    data_cifar100_c: Path = Path('./dataset/cifar100-c')
    data_cifar10_c: Path = Path('./dataset/cifar10-c')


@dataclass
class TrainingConfig:
    optimizer: Literal[
        "SGD",
        "AdamW"
    ] = "SGD"
    lr: float = 0.001
    batch_size: int = 32
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
            'swin_base_patch4_window7_224',
            'convnext_base',
        ] = "vit_base_patch16_224"
    

@dataclass
class DataConfig:
    num_class: int = 1000
    used_data_num: int = -1  # -1 means to use all the data
    resize_only: bool = False
    shuffle: bool = True
    level: Literal[1,2,3,4,5] = 5  # corruption level for corrupted dataset
    corruption: Literal[
        'rendition','sketch','imagenet_a', 'imagenet_c_val_mix','original','imagenet_c_test_mix',
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression', 'speckle_noise', 'spatter', 'gaussian_blur', 'saturate',
        'potpourris','potpourris+', 'cifar10-c','cifar100-c'
    ] = "imagenet_c_val_mix"



@dataclass
class MoETTAConfig:
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
    
    self_router: bool = False
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
    
    dynamic_threshold: bool = False
    """Whether to use dynamic thresholding."""
    
    samplewise: bool = False
    """Whether to route on a per-sample basis."""

    log_matrix_step: int = 100
    """Step interval for logging matrices."""

    disabled_layer: str = ""

    normal_layer: str = ""

    pass_through_coeff: bool = True

    log_detail: bool = False

    early_stop: bool = False
    
    dynamic_lb: bool = False

    fitness_lambda: float = 0.0

    shared_lambda: float = 0.0

    kl_lambda: float = 0.0

    pretrained_router_dir: str = ""

    global_router_idx: float = -1.0

    param_id: str = ""

    param_path: str = ""

    selected_expert: int = -1


@dataclass
class AlgorithmConfig:
    moetta: MoETTAConfig
    
@dataclass
class TuneConfig:
    search_space: Path = ""
    gpu_per_trial: float = 1.0
    cpu_per_trial: float = 4.0
    num_samples: int = 10
    max_t: int = 3600
    job_name: str = ""
    scheduler: Literal[
        "HyperBandScheduler"
    ] = "HyperBandScheduler"
    search_algorithm: Literal[
        "ax",
        "bayes",
        "optuna",
        "basic"
    ] = "optuna"
    mode: Literal["min","max"] = "max"
    metric: str = "overall_accuracy"
    
@dataclass
class Config:
    env: EnvironmentConfig
    train: TrainingConfig
    model: ModelConfig
    data: DataConfig
    algo: AlgorithmConfig
    tune: TuneConfig



