CONFIG = dict()

from .config import Config
from .subconfigs import potpourri, convnext, vit_large

__all__ = ["Config", "CONFIG"]