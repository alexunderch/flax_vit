from typing import Dict, Optional, Tuple
import wandb
import jax.numpy as jnp
from dataclasses import dataclass
import logging
import sys

def make_cli_logger() -> Tuple[int, logging.Logger]:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logging.INFO, logger
@dataclass
class WandbLogger():
    project_name: str
    wandb_config: Dict
    training_config: Dict
    
    def __post_init__(self) -> None:
        wandb.init(
            config = self.training_config,
            project = self.project_name,
            **self.wandb_config
        )
    
    def log_metrics(self, metrics: Dict, 
                          step: int,
                          prefix: Optional[str] = None
                    ) -> None:
        if prefix is not None:
            for k in list(metrics.keys()):
                metrics[f"{prefix}_{k}"] = metrics.pop(k) 

        wandb.log(metrics, step = step)
    
    def log_imagedata(self, images: jnp.ndarray,
                            metadata: Dict
                     ) -> None:
        #TODO: more appropriate logging
        images = wandb.Image(images, caption = metadata["caption"])
        wandb.log({metadata["log_as"]: images})