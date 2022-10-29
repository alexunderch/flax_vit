import sys
sys.path.append("..")
sys.path.append(".")

import jax
import jax.numpy as jnp
from nn.train_main import full_trainining
from datetime import datetime
def main():
    image_size = 224
    patch_size = 32
    print("devices for training:", jax.device_count())
    
    seed = 42
    model_config = dict(
                        n_blocks = 6,
                        block_config = {
                            "latent_dim": 1024 ,
                            "latent_ffd_dim": 1024,
                            "n_heads": 8 ,
                            "dropout_rate_ffd": .1 ,
                            "dropout_rate_att": .1
                        },
                        dropout_embedding = .1,
                        img_params = (image_size, patch_size)
                        ) 

    config = dict(model_config = model_config)

    config.update(
                    dict(weight_decay = .01,
                        learning_rate = 3e-4,
                        warmup_epochs = 10,
                        num_epochs=  35,
                        clip_parameter = 10.,
                        batch_size = 30
                        )
                )

    logger_kwargs = dict(
        project_name = "training-ViT-with-flax",
        wandb_config = dict(
            job_type = "train_and_eval",
            # name = f"run_{datetime.now()}",
            dir = "../wandb",
            # entity = "Sacha"

        )
    )

    dataset_kwargs = dict(
        dataset_name = "caltech101",
        validation_split = .5,
        force_download = True,

    )

    full_trainining(
        config = config,
        seed = seed,
        logger_kwargs = logger_kwargs,
        dataset_kwargs = dataset_kwargs
    )

if __name__ == "__main__":
    main()
