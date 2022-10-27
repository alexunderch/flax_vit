from pathlib import Path
from flax.training import checkpoints
import os
from flax.training.train_state import TrainState
import jax.numpy as jnp
import optax
from dataclasses import dataclass, field
from flax.serialization import (
    to_state_dict, msgpack_serialize, from_bytes
)
import jax
import wandb


def save_checkpoint(state: TrainState,
                    checkpoint_dir: Path) -> None:
    os.makedirs(checkpoint_dir, exist_ok = True)
    checkpoints.save_checkpoint(
        ckpt_dir = checkpoint_dir,
        prefix = "chkpnt_",
        target = state,
        step = state.step,
        overwrite = False, 
        keep = 1
    )

def restore_checkpoint(state: TrainState, 
                      checkpoint_dir: Path) -> TrainState:
    
    assert os.path.isdir(checkpoint_dir)
    return checkpoints.restore_checkpoint(
        ckpt_dir = checkpoint_dir,
        target = state,
        step = state.step,
        prefix = "chkpnt_"
    )


def checkpoint_exists(ckpt_file) -> bool:
    return os.path.isfile(ckpt_file)

####wandb utils
def save_checkpoint_wandb(ckpt_path, state: TrainState, step: int):
    
    artifact = wandb.Artifact(
        f'{wandb.run.name}-checkpoint', type='model'
    )
    with open(ckpt_path, "wb") as outfile:
        outfile.write(msgpack_serialize(to_state_dict(state)))
    artifact.add_file(ckpt_path)
    wandb.log_artifact(artifact, aliases=["latest", f"step_{step}"])


def restore_checkpoint_wandb(ckpt_file, state: TrainState):
    assert checkpoint_exists(ckpt_file)
    artifact = wandb.use_artifact(
                                    f'{wandb.run.name}-checkpoint:latest',
                                 )
    artifact_dir = artifact.download()
    ckpt_path = os.path.join(artifact_dir, ckpt_file)
    with open(ckpt_path, "rb") as data_file:
        byte_data = data_file.read()
    return from_bytes(state, byte_data)

# def hyperparameter_sweep(sweep_config: Dict):
#     wandb.sweep(sweep_config, project = project_name)
####wandb utils


def convert_hidden_state_to_image(input_data: jnp.ndarray, idx: int) -> jnp.ndarray:
    
    """"""
    # input_data = jax.device_get(input_data[idx])
    # attn_maps = [jax.device_get(m[idx]) for m in attn_maps]
    # foreach head
    
    