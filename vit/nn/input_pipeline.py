from distutils.command.config import config
import tensorflow as tf
import tensorflow_datasets as tfds
from utils import Config


config = Config()
(full_train_set, test_dataset), ds_info = tfds.load(
    'cifar10',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.
    return image, label

full_train_set = full_train_set.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE
)

num_data = tf.data.experimental.cardinality(
    full_train_set
).numpy()
print("Total number of data points:", num_data)
train_dataset = full_train_set.take(
    num_data * (1 - config.validation_split)
)
val_dataset = full_train_set.take(
    num_data * (config.validation_split)
)
print(
    "Number of train data points:",
    tf.data.experimental.cardinality(train_dataset).numpy()
)
print(
    "Number of val data points:",
    tf.data.experimental.cardinality(val_dataset).numpy()
)

train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(
    tf.data.experimental.cardinality(train_dataset).numpy()
)
train_dataset = train_dataset.batch(config.batch_size)

val_dataset = val_dataset.cache()
val_dataset = val_dataset.shuffle(
    tf.data.experimental.cardinality(val_dataset).numpy()
)
val_dataset = val_dataset.batch(config.batch_size)


test_dataset = test_dataset.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE
)
print(
    "Number of test data points:",
    tf.data.experimental.cardinality(test_dataset).numpy()
    )
test_dataset = test_dataset.cache()
test_dataset = test_dataset.batch(config.batch_size)