import sys
sys.path.append("..")
sys.path.append(".")

from flax.training.train_state import TrainState
import jax

import jax.numpy as jnp
import optax
from vit.models import VisualTransformer
from nn.train_utils import  create_learning_rate_fn, init_train_state
from nn.train_main import full_trainining

def main():
    image_size = 224
    patch_size = 16
    print(jax.device_count())
    
    model_config = dict(
                        n_blocks = 3,
                        block_config = {
                            "latent_dim": 512 ,
                            "latent_ffd_dim": 1203,
                            "n_heads": 8 ,
                            "dropout_rate_ffd": .1 ,
                            "dropout_rate_att": .5
                        },
                        dropout_embedding = .5,
                        img_params = (image_size, patch_size)
                        ) 
    seed = 42

    config = dict(model_config = model_config)

    config.update(
                    dict(weight_decay = .001,
                        learning_rate = 1e-5,
                        warmup_epochs = 1,
                        num_epochs=  3,
                        clip_parameter = 1.,
                        batch_size = 1
                        )
                )

    logger_kwargs = dict(
        project_name = "training ViT with flax",
        wandb_config = dict()
    )

    dataset_kwargs = dict(
        dataset_name = "caltech101",
        validation_split = .3,
        force_download = False,

    )

    full_trainining(
        config = config,
        seed = seed,
        logger_kwargs = logger_kwargs,
        dataset_kwargs = dataset_kwargs
    )

if __name__ == "__main__":
    main()
