import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
import torch.utils.data
from torch.utils.data.dataset import Subset
import timm
from loguru import logger

from ..config import Config
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


def get_data(corruption, config: Config):
    model = timm.create_model(config.model.model, pretrained=False)
    normalize = transforms.Normalize(
        mean=timm.data.resolve_data_config({}, model=model)["mean"],
        std=timm.data.resolve_data_config({}, model=model)["std"],
    )
    del model
    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transforms_imagenet_C = transforms.Compose(
        [transforms.CenterCrop(224), transforms.ToTensor(), normalize]
    )

    match corruption:
        case "original":
            test_set = ImageFolder(
                root=os.path.join(os.path.expanduser(config.env.data), "val"),
                transform=test_transforms,
            )
        case corruption if corruption in COMMON_CORRUPTIONS:
            test_set = ImageFolder(
                root=os.path.join(
                    os.path.expanduser(config.env.data_corruption),
                    corruption,
                    str(config.data.level),
                ),
                transform=test_transforms_imagenet_C,
            )
        case "rendition":
            test_set = ImageFolder(
                root=os.path.expanduser(config.env.data_rendition),
                transform=test_transforms,
                target_transform=lambda idx: r_to_origin[idx],
            )
        case "sketch":
            test_set = datasets.ImageFolder(
                root=os.path.expanduser(config.env.data_sketch),
                transform=test_transforms,
            )
        case "imagenet_a":
            test_set = datasets.ImageFolder(
                root=os.path.expanduser(config.env.data_adv),
                transform=test_transforms,
                target_transform=lambda idx: a_to_origin[idx],
            )

    return test_set


def prepare_test_data(config: Config):
    match config.data.corruption:
        case "original" | "rendition" | "sketch" | "imagenet_a":
            test_set = get_data(config.data.corruption, config)
        case corruption if corruption in COMMON_CORRUPTIONS:
            test_set = get_data(corruption, config)
        case "imagenet_c_test_mix":
            dataset_list = [
                get_data(corruption, config) for corruption in COMMON_CORRUPTIONS_15
            ]
            test_set = torch.utils.data.ConcatDataset(dataset_list)  # 合并多个dataset
            if config.data.used_data_num != -1:
                logger.info(
                    f"Creating subset of {config.data.used_data_num} samples from imagenet_c_test_mix"
                )
                test_set = Subset(
                    test_set, torch.randperm(len(test_set))[: config.data.used_data_num]
                )
        case "imagenet_c_val_mix":
            dataset_list = [
                get_data(corruption, config) for corruption in COMMON_CORRUPTIONS_4
            ]
            test_set = torch.utils.data.ConcatDataset(dataset_list)  # 合并多个dataset
            if config.data.used_data_num != -1:
                # create subset
                logger.info(
                    f"Creating subset of {config.data.used_data_num} samples from imagenet_c_val_mix"
                )
                test_set = Subset(
                    test_set, torch.randperm(len(test_set))[: config.data.used_data_num]
                )
        case "potpourris":
            dataset_list = [
                get_data(corruption, config) for corruption in COMMON_CORRUPTIONS_15
            ]
            dataset_list.append(get_data("rendition", config))
            dataset_list.append(get_data("sketch", config))
            dataset_list.append(get_data("imagenet_a", config))
            test_set = torch.utils.data.ConcatDataset(dataset_list)  # 合并多个dataset
        case "potpourris+":
            dataset_list = [
                get_data(corruption, config) for corruption in COMMON_CORRUPTIONS_15
            ]
            dataset_list.append(get_data("rendition", config))
            dataset_list.append(get_data("sketch", config))
            dataset_list.append(get_data("imagenet_a", config))
            dataset_list.append(get_data("original", config))
            test_set = torch.utils.data.ConcatDataset(dataset_list)  # 合并多个dataset
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
