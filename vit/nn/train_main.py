from typing import Dict, List
from .train_utils import train_step, eval_step, accumulate_metrics
from .input_pipeline import prepare_data
from .common_utils import (convert_hidden_state_to_image, 
                           save_checkpoint_wandb, 
                           restore_checkpoint_wandb)
import jax, flax
from tqdm import tqdm
from .wandb_logger import WandbLogger
from flax.training.train_state import TrainState

def full_trainining(
    datasets: List,
    initial_state: TrainState,
    num_iterations,
    loogging_freq: int,
    config: Dict,
    logger: WandbLogger,
):
    train_dataset, eval_dataset, test_dataset = datasets

    for n_iter in tqdm(range(1, num_iterations + 1)):
        batch_metrics = dict(
                                train = [],
                                eval = []
                            ) 

        
                            


def train_and_evaluate(
    train_dataset,
    eval_dataset,
    test_dataset,
    state: TrainState,
    num_iterations: int,
    logging_freq: int
):

    for epoch in tqdm(range(1, num_iterations + 1)):

        best_eval_loss = 1e6
        
        train_batch_metrics = []
        train_datagen = iter(tfds.as_numpy(train_dataset))
        for batch_idx in range(num_train_batches):
            batch = next(train_datagen)
            state, metrics = train_step(state, batch)
            train_batch_metrics.append(metrics)
        
        train_batch_metrics = accumulate_metrics(train_batch_metrics)
        print(
            'TRAIN (%d/%d): Loss: %.4f, accuracy: %.2f' % (
                epoch, epochs, train_batch_metrics['loss'],
                train_batch_metrics['accuracy'] * 100
            )
        )

        eval_batch_metrics = []
        eval_datagen = iter(tfds.as_numpy(eval_dataset))
        for batch_idx in range(num_eval_batches):
            batch = next(eval_datagen)
            metrics = eval_step(state, batch)
            eval_batch_metrics.append(metrics)
        
        eval_batch_metrics = accumulate_metrics(eval_batch_metrics)
        print(
            'EVAL (%d/%d):  Loss: %.4f, accuracy: %.2f\n' % (
                epoch, epochs, eval_batch_metrics['loss'],
                eval_batch_metrics['accuracy'] * 100
            )
        )

        wandb.log({
            "Train Loss": train_batch_metrics['loss'],
            "Train Accuracy": train_batch_metrics['accuracy'],
            "Validation Loss": eval_batch_metrics['loss'],
            "Validation Accuracy": eval_batch_metrics['accuracy']
        }, step=epoch)

        if eval_batch_metrics['loss'] < best_eval_loss:
            save_checkpoint("checkpoint.msgpack", state, epoch)
    
    restored_state = load_checkpoint("checkpoint.msgpack", state)
    test_batch_metrics = []
    test_datagen = iter(tfds.as_numpy(test_dataset))
    for batch_idx in range(num_test_batches):
        batch = next(test_datagen)
        metrics = eval_step(restored_state, batch)
        test_batch_metrics.append(metrics)
    
    test_batch_metrics = accumulate_metrics(test_batch_metrics)
    print(
        'Test: Loss: %.4f, accuracy: %.2f' % (
            test_batch_metrics['loss'],
            test_batch_metrics['accuracy'] * 100
        )
    )

    wandb.log({
        "Test Loss": test_batch_metrics['loss'],
        "Test Accuracy": test_batch_metrics['accuracy']
    })
    
    return state, restored_state