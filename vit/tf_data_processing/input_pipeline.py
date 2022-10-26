import os
from typing import List, Optional, Tuple, Any

import flax
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds



def prepare_data(dataset_name: str,
                validation_split: float,
                batch_size: int,
                image_size: int,
                force_download: bool = False
                ) -> Tuple:

    (train_dataset, full_test_dataset), ds_info = tfds.load(
                                                                dataset_name,
                                                                split=['train', 'test'],
                                                                shuffle_files = False,
                                                                as_supervised = True,
                                                                with_info = True,
                                                                download = force_download
                                                            )

    print(f"Got a dataset with info:\n{ds_info}")

    def normalize__and_resize_img(image, label):
        image = tf.image.resize(image, size = (image_size, image_size))
        image = tf.cast(image, tf.float32) / 255.
        return image, label

    train_dataset = train_dataset.map(
        normalize__and_resize_img, num_parallel_calls=tf.data.AUTOTUNE
    )

    train_dataset = train_dataset.shuffle(
                                            len(train_dataset)
                                        #   tf.data.experimental.cardinality(train_dataset).numpy(),
                                         )


    full_test_dataset = full_test_dataset.map(
                                                normalize__and_resize_img, num_parallel_calls=tf.data.AUTOTUNE
                                             )
    num_data = tf.data.experimental.cardinality(
                                                full_test_dataset
                                                ).numpy()
    eval_dataset = full_test_dataset.take(
                                        num_data * (validation_split)
                                        ).cache().repeat()  

    test_dataset = full_test_dataset.take(
                                        num_data * (1. - validation_split)
                                        ).cache().repeat()           


    eval_dataset = eval_dataset.shuffle(
                                        num_data * (validation_split), 
                                       )

    train_dataset = train_dataset.cache().batch(batch_size)
    eval_dataset = eval_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    print(
        "Number of train data points:",
        len(train_dataset)
        )
    print(
        "Number of eval data points:",
        num_data * (validation_split)
        )
    print(
        "Number of test data points:",
         num_data * (1. - validation_split)                            
        )
    
    return train_dataset, eval_dataset, test_dataset, ds_info

def get_classes(ds_info: Any) -> List[str]:
    return ds_info.features['label'].names

def prefetch(iterable_dataset, n_prefetch: Optional[int] = None):
    iterable_dataset = map(lambda x: jax.tree_map(lambda t: np.asarray(memoryview(t)), x),
                iterable_dataset)
    if n_prefetch:
        iterable_dataset = flax.jax_utils.prefetch_to_device(iterable_dataset, n_prefetch)
    return iterable_dataset
