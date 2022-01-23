# coding:utf-8

import tensorflow_datasets as tfds
import tensorflow as tf

(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    as_supervised=True,
    with_info=True
)


