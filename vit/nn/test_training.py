import sys


sys.path.append("..")
sys.path.append(".")

from flax.training.train_state import TrainState
import jax

import jax.numpy as jnp
import optax
from vit.models import VisualTransformer
from train_utils import train_step, eval_step, create_learning_rate_fn

import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'

def test_step():
    print(jax.devices("cpu"))
    rng = jax.random.PRNGKey(0)
    _, dropout_rng = jax.random.split(rng)
    model = VisualTransformer(n_blocks = 3,
                               block_config = {
                                    "latent_dim": 512 ,
                                    "latent_ffd_dim": 1203,
                                    "n_heads": 8 ,
                                    "training": True ,
                                    "dropout_rate_ffd": .1 ,
                                    "dropout_rate_att": .5
                                },
                                output_dim = 2,
                                dropout_embedding = .5,
                                img_params = (32, 16))

    
    test_config = dict(
                        weight_decay = .001,
                        lr = 1e-5,
                        warmup_epochs = 3,
                        num_epochs=  1,
                        num_classes = 2)

    lr_schedule = create_learning_rate_fn(config= test_config,
                                          base_learning_rate = 1e-5,
                                          steps_per_epoch = 2)

    test_batch = {
        "image": jnp.zeros((2, 32, 32, 3)),
        "label": jnp.zeros((2, 1))
    }
    params = model.init(rng, test_batch['image'])["params"]


    test_state = TrainState.create(apply_fn=model.apply,
                                   params=params,
                                   tx= optax.adam(learning_rate = 1e-5))

    print(
        train_step(state = test_state, 
               batch = test_batch, 
               learning_rate_fn = lr_schedule,
               num_classes = 2, 
               config = test_config,
               dropout_rng = dropout_rng)[1]
    )

    print(
        eval_step(state = test_state, 
               batch = test_batch, 
               num_classes = 2, 
               dropout_rng = dropout_rng)
    )


if __name__ == "__main__":
    test_step()