from ..config import Config, CONFIG
from dataclasses import replace

config_base = Config()
config_convnext = replace(
    config_base,
    model=replace(config_base.model, model="convnext_base"),
    algo=replace(
        config_base.algo,
        moetta=replace(
            config_base.algo.moetta,
            disabled_layer="0-3",
        ),
    )
)
CONFIG["convnext"] = ("Config for convnext model", config_convnext)