from .config import Config
from .utils import wandb_log

@wandb_log
def pipeline(config:Config):
    return print("running pipeline!")