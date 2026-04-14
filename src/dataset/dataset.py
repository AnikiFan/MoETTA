import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
import torch.utils.data
from torch.utils.data.dataset import Subset
from loguru import logger
from PIL import Image

from config import Config
from src.model_utils import create_model_from_config
from .ImageNetMask import r_to_origin, a_to_origin

COMMON_CORRUPTIONS_15 = [
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
]

COMMON_CORRUPTIONS_4 = [
    "speckle_noise",
    "spatter",
    "gaussian_blur",
    "saturate",
]

COMMON_CORRUPTIONS = COMMON_CORRUPTIONS_15 + COMMON_CORRUPTIONS_4


def resolve_dataset_family(config: Config, corruption: str) -> str:
    if config.data.dataset != "auto":
        return config.data.dataset
    corruption_to_dataset = {
        "cifar10-c": "cifar10",
        "cifar100-c": "cifar100",
    }
    return corruption_to_dataset.get(
        corruption,
        corruption_to_dataset.get(config.data.corruption, "imagenet"),
    )


class CIFARCorruptionDataset(torch.utils.data.Dataset):
    def __init__(self, root, corruption: str, severity: int, transform=None):
        self.root = Path(root).expanduser()
        self.corruption = corruption
        self.severity = severity
        self.transform = transform

        corruption_file = self.root / f"{corruption}.npy"
        labels_file = self.root / "labels.npy"
        if not corruption_file.exists():
            raise FileNotFoundError(
                f"Cannot find corruption file: {corruption_file}"
            )
        if not labels_file.exists():
            raise FileNotFoundError(f"Cannot find labels file: {labels_file}")

        self.data = np.load(corruption_file, mmap_mode="r")
        labels = np.load(labels_file)
        if len(labels) % 5 != 0:
            raise ValueError(
                f"Unexpected CIFAR-C label count {len(labels)} in {labels_file}"
            )
        samples_per_level = len(labels) // 5
        start = (severity - 1) * samples_per_level
        end = severity * samples_per_level
        self.targets = labels[start:end]
        self.data = self.data[start:end]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.fromarray(np.asarray(self.data[index]))
        target = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_transforms(config: Config):
    model = create_model_from_config(config, pretrained=False)
    data_config = model.pretrained_cfg
    if not data_config:
        data_config = model.default_cfg
    normalize = transforms.Normalize(
        mean=data_config["mean"],
        std=data_config["std"],
    )
    del model
    crop_size = data_config["input_size"][-2:]
    resize_size = tuple(round(size * 256 / 224) for size in crop_size)
    imagenet_transforms = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    imagenet_c_transforms = transforms.Compose(
        [transforms.CenterCrop(crop_size), transforms.ToTensor(), normalize]
    )
    cifar_transforms = transforms.Compose(
        [
            transforms.Resize(crop_size, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return imagenet_transforms, imagenet_c_transforms, cifar_transforms


def build_cifar_dataset(config: Config, dataset_name: str, transform):
    dataset_root = Path(getattr(config.env, f"{dataset_name}_data_path")).expanduser()
    dataset_cls = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100
    return dataset_cls(
        root=dataset_root, train=False, download=False, transform=transform
    )


def get_data(corruption, config: Config):
    dataset_family = resolve_dataset_family(config, corruption)
    test_transforms, test_transforms_imagenet_C, test_transforms_cifar = (
        build_transforms(config)
    )

    match corruption:
        case "original":
            match dataset_family:
                case "imagenet":
                    test_set = ImageFolder(
                        root=os.path.join(
                            os.path.expanduser(config.env.original_data_path), "val"
                        ),
                        transform=test_transforms,
                    )
                case "cifar10" | "cifar100":
                    test_set = build_cifar_dataset(config, dataset_family, test_transforms_cifar)
                case _:
                    raise ValueError(f"Unsupported dataset family: {dataset_family}")
        case corruption if corruption in COMMON_CORRUPTIONS:
            match dataset_family:
                case "imagenet":
                    test_set = ImageFolder(
                        root=os.path.join(
                            os.path.expanduser(config.env.corruption_data_path),
                            corruption,
                            str(config.data.level),
                        ),
                        transform=test_transforms_imagenet_C,
                    )
                case "cifar10" | "cifar100":
                    if corruption not in COMMON_CORRUPTIONS_15:
                        raise ValueError(
                            f"CIFAR-C does not provide corruption `{corruption}`."
                        )
                    dataset_root = Path(
                        getattr(config.env, f"{dataset_family}_c_path")
                    ).expanduser()
                    test_set = CIFARCorruptionDataset(
                        root=dataset_root,
                        corruption=corruption,
                        severity=config.data.level,
                        transform=test_transforms_cifar,
                    )
                case _:
                    raise ValueError(
                        f"Unsupported dataset family `{dataset_family}`."
                    )
        case "rendition":
            if dataset_family != "imagenet":
                raise ValueError("ImageNet-R only supports dataset=imagenet.")
            test_set = ImageFolder(
                root=os.path.expanduser(config.env.rendition_data_path),
                transform=test_transforms,
                target_transform=lambda idx: r_to_origin[idx],
            )
        case "sketch":
            if dataset_family != "imagenet":
                raise ValueError("ImageNet-Sketch only supports dataset=imagenet.")
            test_set = datasets.ImageFolder(
                root=os.path.expanduser(config.env.sketch_data_path),
                transform=test_transforms,
            )
        case "imagenet_a":
            if dataset_family != "imagenet":
                raise ValueError("ImageNet-A only supports dataset=imagenet.")
            test_set = datasets.ImageFolder(
                root=os.path.expanduser(config.env.adv_data_path),
                transform=test_transforms,
                target_transform=lambda idx: a_to_origin[idx],
            )
        case "cifar10-c" | "cifar100-c":
            dataset_name = corruption.removesuffix("-c")
            dataset_root = Path(
                getattr(config.env, f"{dataset_name}_c_path")
            ).expanduser()
            test_set = torch.utils.data.ConcatDataset(
                [
                    CIFARCorruptionDataset(
                        root=dataset_root,
                        corruption=cifar_corruption,
                        severity=config.data.level,
                        transform=test_transforms_cifar,
                    )
                    for cifar_corruption in COMMON_CORRUPTIONS_15
                ]
            )
        case _:
            raise ValueError(f"Corruption `{corruption}` not found!")

    return test_set


def prepare_test_data(config: Config):
    def maybe_subset_dataset(dataset, dataset_name: str):
        if config.data.used_data_num == -1:
            return dataset
        logger.info(
            f"Creating subset of {config.data.used_data_num} samples from {dataset_name}"
        )
        return Subset(dataset, torch.randperm(len(dataset))[: config.data.used_data_num])

    match config.data.corruption:
        case (
            "original"
            | "rendition"
            | "sketch"
            | "imagenet_a"
            | "cifar10-c"
            | "cifar100-c"
        ):
            test_set = maybe_subset_dataset(
                get_data(config.data.corruption, config), config.data.corruption
            )
        case corruption if corruption in COMMON_CORRUPTIONS:
            test_set = maybe_subset_dataset(get_data(corruption, config), corruption)
        case "imagenet_c_test_mix":
            dataset_list = [
                get_data(corruption, config) for corruption in COMMON_CORRUPTIONS_15
            ]
            test_set = maybe_subset_dataset(
                torch.utils.data.ConcatDataset(dataset_list), "imagenet_c_test_mix"
            )
        case "imagenet_c_val_mix":
            dataset_list = [
                get_data(corruption, config) for corruption in COMMON_CORRUPTIONS_4
            ]
            test_set = maybe_subset_dataset(
                torch.utils.data.ConcatDataset(dataset_list), "imagenet_c_val_mix"
            )
        case "potpourri":
            dataset_list = [
                get_data(corruption, config) for corruption in COMMON_CORRUPTIONS_15
            ]
            dataset_list.append(get_data("rendition", config))
            dataset_list.append(get_data("sketch", config))
            dataset_list.append(get_data("imagenet_a", config))
            test_set = maybe_subset_dataset(
                torch.utils.data.ConcatDataset(dataset_list), "potpourri"
            )
        case "potpourri+":
            dataset_list = [
                get_data(corruption, config) for corruption in COMMON_CORRUPTIONS_15
            ]
            dataset_list.append(get_data("rendition", config))
            dataset_list.append(get_data("sketch", config))
            dataset_list.append(get_data("imagenet_a", config))
            dataset_list.append(get_data("original", config))
            test_set = maybe_subset_dataset(
                torch.utils.data.ConcatDataset(dataset_list), "potpourri+"
            )
        case _:
            raise ValueError("Corruption not found!")

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.train.batch_size,
        shuffle=config.data.shuffle,
        num_workers=config.train.workers,
        pin_memory=True,
    )
    return test_set, test_loader
