import sys
from typing import Dict
sys.path.append("..")
sys.path.append(".")

import jax
import jax.numpy as jnp
from nn.train_main import full_trainining
import wandb
# from datetime import datetime

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
                        num_epochs=  3,
                        clip_parameter = 10.,
                        batch_size = 40
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

def hyperparameter_sweep(
                        project_name: str,
                        parameters_dict: Dict
                        ):
    sweep_configuration = {
        'method': 'bayes',
        'name': 'SWEEP',
        'metric': {'goal': 'maximize', 'name': 'test_f1'},
        'parameters': parameters_dict
    }

    sweep_id = wandb.sweep(sweep = sweep_configuration, project = project_name)
    return sweep_id


if __name__ == "__main__":
    parameters_dict =  {
        # 'batch_size': {'values': [16, 32, 64]},
        'num_epochs': {'values': [20, 40, 60]},
        'learning_rate': {'max': 0.1, 'min': 1e-4},
        'weight_decay': {'max': 0.1, 'min': 1e-4},
        'clip_parameter': {'max': 10., 'min': .1},
        'n_blocks': {'values': [4, 8, 12]}

    }
    sweep_id = hyperparameter_sweep("training-ViT-with-flax", 
                                    parameters_dict)
    wandb.agent(sweep_id, function = main, count = 4)