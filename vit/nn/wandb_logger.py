from typing import Dict
import wandb
import jax.numpy as jnp
from dataclasses import dataclass

@dataclass
class WandbLogger():
    project_name: str
    wandb_config: Dict
    training_config: Dict
    seed: int

    def __post_init__(self) -> None:
        wandb.init(
            config = self.wandb_config,
            project = self.project_name
        )
    
    def log_metrics(self, metrics: Dict, 
                          step: int,
                    ) -> None:
        wandb.log(metrics, step = step)
    
    def log_imagedata(self, images: jnp.ndarray,
                            metadata: Dict
                     ) -> None:
        images = wandb.Image(images, caption = metadata["caption"])
        wandb.log({metadata["log_as"]: images})