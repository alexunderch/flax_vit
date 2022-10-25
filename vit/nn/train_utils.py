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

from typing import Dict, Callable, Tuple


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray, num_classes: int):
    one_hot_labels = jax.nn.one_hot(labels, num_classes = num_classes)
    xentropy = optax.softmax_cross_entropy(logits = logits, 
                                        labels = one_hot_labels)
    return jnp.mean(xentropy)

def accumulate_metrics(metrics):
    metrics = jax.device_get(metrics)
    return {
        k: jnp.mean([metric[k] for metric in metrics])
        for k in metrics[0]
    }

def compute_metrics(logits: jnp.ndarray, labels: jnp.ndarray, num_classes: int):
    loss = cross_entropy_loss(logits, labels, num_classes)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
   
    def f1(y_true, y_predicted, label):

        tp = jnp.sum((y_true == label) & (y_predicted == label))
        fp = jnp.sum((y_true != label) & (y_predicted == label))
        fn = jnp.sum((y_predicted != label) & (y_true == label))
        
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)

        return 2 * (precision * recall) / (precision + recall)
        

    metrics = {
        'loss': loss.mean(),
        'accuracy': accuracy.mean(),
        'f1': jnp.array([f1(labels, logits, l) for l in range(num_classes)]).mean()
    }
    #mean across all devices batches
    # metrics = lax.pmean(metrics, axis_name = 'batch')
    return metrics


def create_learning_rate_fn(
                            config: Dict,
                            base_learning_rate: float,
                            steps_per_epoch: int) -> Callable:
    warmup_fn = optax.linear_schedule(
                                    init_value = 0., end_value = base_learning_rate,
                                    transition_steps = config["warmup_epochs"] * steps_per_epoch
                                    )
    cosine_epochs = max(config["num_epochs"] - config["warmup_epochs"], 1)
    cosine_fn = optax.cosine_decay_schedule(
                                            init_value=base_learning_rate,
                                            decay_steps=cosine_epochs * steps_per_epoch
                                        )
    schedule_fn = optax.join_schedules(
                                    schedules=[warmup_fn, cosine_fn],
                                    boundaries=[config["warmup_epochs"] * steps_per_epoch]
                                    )
    return schedule_fn


@functools.partial(jax.jit, static_argnums=(2, 3))
def train_step(state: TrainState, 
               batch: Dict, 
               learning_rate_fn: Callable,
               num_classes: int,
               config: Dict,
               dropout_rng = None) -> Tuple[TrainState, Dict]:

    def loss_fn(params):
        logits = state.apply_fn(dict(params = params), 
                                batch['image'],
                                rngs = dict(dropout= dropout_rng))                       
        loss = cross_entropy_loss(logits, batch['label'], num_classes)
        weight_penalty_params = jax.tree_util.tree_leaves(params)
        weight_l2 = sum(jnp.sum(x ** 2) 
                        for x in weight_penalty_params 
                        if x.ndim > 1)
    
        loss = loss + config["weight_decay"] * weight_l2
        return loss, logits

    lr = learning_rate_fn(state.step)
    grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
    (_, logits), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads = grads)

    metrics = compute_metrics(logits, batch['label'], num_classes)
    metrics['learning_rate'] = lr

    return new_state, metrics

@functools.partial(jax.jit, static_argnums = 2)
def eval_step(state: TrainState, 
              batch: Dict,
              num_classes: int,
              dropout_rng = None,
              ) -> Dict:
    logits = state.apply_fn(dict(params = state.params), 
                            batch['image'], 
                            rngs = dict(dropout = dropout_rng))
    return compute_metrics(logits, batch['label'], num_classes)

