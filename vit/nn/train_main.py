import sys
sys.path.append("..")
from typing import Dict, List, Optional, Tuple
from train_utils import (pmap_steps, 
                        accumulate_metrics, 
                        create_learning_rate_fn, 
                        init_train_state
                        )
from tf_data_processing.input_pipeline import prepare_data, get_classes, prefetch
from common_utils import (convert_hidden_state_to_image, 
                          save_checkpoint_wandb, 
                          restore_checkpoint_wandb,
                          checkpoint_exists
                          )
from vit.models import VisualTransformer
import jax, flax
from tqdm import tqdm
import jax.numpy as jnp
from wandb_logger import WandbLogger, make_cli_logger
from flax.training.train_state import TrainState
import tensorflow_datasets as tfds


def full_trainining(
                    config: Dict,
                    seed: int,
                    dataset_kwargs: Dict,
                    logger_kwargs: Dict,
                    state: Optional[TrainState] = None,
                    logger: WandbLogger = WandbLogger,
                    ) -> Tuple[TrainState, TrainState]:

    rng = jax.random.PRNGKey(seed = seed)
    _, dropout_rng = jax.random.split(rng)
    n_prefetch = jax.device_count()
    train_dataset, eval_dataset, test_dataset, ds_info = prepare_data(
                                                                      batch_size = config["batch_size"],
                                                                      image_size = config["model_config"]["img_params"][0],
                                                                      **dataset_kwargs,
                                                                     )
    num_classes = len(get_classes(ds_info))
    batch = next(iter(tfds.as_numpy(train_dataset)))
    config["model_config"].update({"output_dim": num_classes})

    if state is None:
        state = init_train_state(
                                model = VisualTransformer(**config["model_config"]), 
                                random_key = rng, 
                                shape = batch[0].shape,
                                learning_rate = config["learning_rate"],
                                clip_parameter = config["clip_parameter"],
                                )
    del batch
    level, cli_logger = make_cli_logger()
    wandb_logger = logger(
                          training_config = config,
                          **logger_kwargs
                          )
    #TODO: replicas, pmap
    train_step, eval_step = pmap_steps()
    # state = flax.jax_utils.replicate(state)

    for epoch in tqdm(range(1, config["num_epochs"] + 1)):
        best_epoch_eval_loss = jnp.inf
        batch_metrics = dict(
                            train = [],
                            eval = [],
                            test = []
                            )

        #### training phase ####
        for batch in iter(tfds.as_numpy(train_dataset)):
            batch = dict(
                        image = batch[0],
                        label = batch[1]
                        )
            state, metrics = train_step(
                                        state = state,
                                        batch = batch,
                                        learning_rate_fn = create_learning_rate_fn(config, 
                                                                                   base_learning_rate = config["learning_rate"],
                                                                                   steps_per_epoch = len(train_dataset)),
                                        num_classes = num_classes, 
                                        config = config,
                                        dropout_rng = dropout_rng
                                        )
            batch_metrics["train"].append(metrics)

        batch_metrics["train"] = accumulate_metrics(batch_metrics["train"])

        cli_logger.log(level, "train logs:\n" + "\n".join([f"{k}:{v}" for k, v in batch_metrics["train"].items()]))
        wandb_logger.log_metrics(batch_metrics["train"], step = epoch, prefix = "train")

        
        #### evaluation phase ####
        for batch in iter(tfds.as_numpy(eval_dataset)):
            batch = dict(
                        image = batch[0],
                        label = batch[1]
                        )
            metrics = eval_step( 
                                state = state, 
                                batch = batch, 
                                num_classes = num_classes, 
                                dropout_rng = dropout_rng
                                )
            batch_metrics["eval"].append(metrics)

        batch_metrics["eval"] = accumulate_metrics(batch_metrics["eval"])
        cli_logger.log(level, "eval logs:\n" + "\n".join([f"{k}:{v}" for k, v in batch_metrics["eval"].items()]))
        wandb_logger.log_metrics(batch_metrics["eval"], step = epoch, prefix = "eval")


        if batch_metrics["eval"]["loss"] < best_epoch_eval_loss:
            best_epoch_eval_loss = batch_metrics["eval"]["loss"]
            # save_checkpoint_wandb()


        #### testing phase ####
        restored_best_state = state #restore_checkpoint_wandb()
        for batch in iter(tfds.as_numpy(test_dataset)):
            batch = dict(
                        image = batch[0],
                        label = batch[1]
                        )
            metrics =  eval_step( 
                                state = restored_best_state, 
                                batch = batch, 
                                num_classes = len(get_classes(...)), 
                                dropout_rng = dropout_rng
                                )
            batch_metrics["test"].append(metrics)

        batch_metrics["test"] = accumulate_metrics(batch_metrics["test"])
        cli_logger.log(level, "test logs:\n" + "\n".join([f"{k}:{v}" for k, v in batch_metrics["test"].items()]))
        logger.log_metrics(batch_metrics["test"], step = epoch, prefix = "test")

    return state, restored_best_state

        
                            
