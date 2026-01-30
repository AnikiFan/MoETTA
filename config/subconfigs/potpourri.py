from ..config import Config, CONFIG
from dataclasses import replace

config_base = Config()
config_potpourri = replace(
    config_base,
    data=replace(config_base.data, corruption="potpourri")
)
CONFIG["potpourri"] = ("Config for potpourri corruption", config_potpourri)