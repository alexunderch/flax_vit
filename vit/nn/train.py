import jax.numpy as jnp
import jax.random as random
from jax import lax
import functools
import flax
import optax
import jax
from flax.training.train_state import TrainState
from utils import Config
from sklearn.metrics import f1_score
from tqdm import tqdm

from typing import Dict, Callable

def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray, num_classes: int):
    one_hot_labels = jax.nn.one_hot(labels, num_classes = num_classes)
    xentropy = optax.softmax_cross_entropy(logits = logits, 
                                        labels = one_hot_labels)
    return jnp.mean(xentropy)


def compute_metrics(logits: jnp.ndarray, labels: jnp.ndarray, log_to_wandb: bool = False):
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    f1 = jnp.array(f1_score(labels, jnp.argmax(logits, -1), average = "weighted"))
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'f1': f1
    }
    #mean across all devices batches
    metrics = lax.pmean(metrics, axis_name = 'batch')
    return metrics


def create_learning_rate_fn(
                            config: Dict,
                            base_learning_rate: float,
                            steps_per_epoch: int) -> Callable:
    warmup_fn = optax.linear_schedule(
                                    init_value = 0., end_value = base_learning_rate,
                                    transition_steps = config.warmup_epochs * steps_per_epoch
                                    )
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
                                            init_value=base_learning_rate,
                                            decay_steps=cosine_epochs * steps_per_epoch
                                        )
    schedule_fn = optax.join_schedules(
                                    schedules=[warmup_fn, cosine_fn],
                                    boundaries=[config.warmup_epochs * steps_per_epoch]
                                    )
    return schedule_fn

@jax.jit
def train_step(state: TrainState, 
               batch: Dict, 
               learning_rate_fn: Callable,
               config: Config,
               dropout_rng = None):
    def loss_fn(params):
        logits = state.apply_fn(dict(params = params), 
                                inputs = batch['image'], 
                                train = True,
                                rngs = dict(dropout= dropout_rng))
        loss = cross_entropy_loss(logits, batch['label'])
        weight_penalty_params = jax.tree_util.tree_leaves(params)
        weight_l2 = sum(jnp.sum(x ** 2) 
                        for x in weight_penalty_params 
                        if x.ndim > 1)
    
        loss = loss + config.weight_decay * weight_l2
        return loss, logits

    lr = learning_rate_fn(state.step)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, "batch")
    new_state = state.apply_gradients(grads = grads)

    metrics = compute_metrics(logits, batch['label'])
    metrics['learning_rate'] = lr

    return new_state, metrics

@jax.jit
def eval_step(state: TrainState, batch, dropout_rng = None):
    inputs, labels = batch['image'], batch['label']
    logits = state.apply_fn(dict(params = state.params), 
                            inputs = inputs, 
                            train = False,
                            rngs = dict(dropout = dropout_rng))
    return compute_metrics(logits, labels)

def accumulate_metrics(metrics):
    metrics = jax.device_get(metrics)
    return {
        k: jnp.mean([metric[k] for metric in metrics])
        for k in metrics[0]
    }



def train_and_evaluate(
    train_dataset,
    eval_dataset,
    test_dataset,
    state: TrainState,
    epochs: int,
):
    # num_train_batches = tf.data.experimental.cardinality(train_dataset)
    # num_eval_batches = tf.data.experimental.cardinality(eval_dataset)
    # num_test_batches = tf.data.experimental.cardinality(test_dataset)
    
    for epoch in tqdm(range(1, epochs + 1)):

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