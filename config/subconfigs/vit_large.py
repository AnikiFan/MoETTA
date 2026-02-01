from ..config import Config, CONFIG
from dataclasses import replace

config_base = Config()
config_vit_large = replace(
    config_base,
    model=replace(config_base.model, model="vit_large_patch16_224"),
    algo=replace(
        config_base.algo,
        moetta=replace(
            config_base.algo.moetta,
            disabled_layer="0-14",
        ),
    )
)
CONFIG["vit_large"] = ("Config for vit_large model", config_vit_large)