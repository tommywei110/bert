# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Text

import absl
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft
import tensorflow_hub as hub
import tensorflow_text
from tensorflow_text.python.ops import bert_tokenizer

from tfx.components.trainer.executor import TrainerFnArgs

_TRAIN_BATCH_SIZE = 512
_TRAIN_DATA_SIZE = 51200
_EVAL_BATCH_SIZE = 512
_LABEL_KEY = "sentiment"

def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP'
          )

def _sentiment_to_int(sentiment):
    """Converting labels from string to integer"""
    equality = tf.equal(sentiment, 'positive')
    ints = tf.cast(equality, tf.int64)
    return ints

def _tokenize(stringA, stringB):
    """Tokenize the two sentences and insert appropriate tokens"""
    tokenizer = bert_tokenizer.BertTokenizer(
            "/home/tommywei/bert_mrpc/vocab.txt",
            token_out_type=tf.string,
            )
    stringA = tf.squeeze(stringA)
    idA = tokenizer.tokenize(stringA)
    #idB = tokenizer.tokenize(stringB)
    return idA

def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  stringA = inputs['stringA']
  stringB = inputs['stringB']
  label = inputs['Quality']
  return {
          'label': label,
          'feature': _tokenize(stringA, stringB)
          }

def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      label_key=_LABEL_KEY)

  return dataset

def _build_keras_model() -> tf.keras.Model:
  """Creates a DNN Keras model for classifying imdb data.

  Returns:
    A Keras Model.
  """
  # The model below is built with Functional API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50-with-normalization/1", output_shape=[50],
                           input_shape=[], dtype=tf.string)

  model = keras.Sequential()
  model.add(hub_layer)
  model.add(keras.layers.Dense(16, activation='relu'))
  model.add(keras.layers.Dense(1, activation='sigmoid'))
  model.compile(optimizer='adam',
          loss=keras.losses.BinaryCrossentropy(from_logits=False),
          metrics=['accuracy'])
  model.summary()
  return model

# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(fn_args.train_files, tf_transform_output,
                            batch_size=_TRAIN_BATCH_SIZE)
  eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output,
                           batch_size=_EVAL_BATCH_SIZE)

  #mirrored_strategy = tf.distribute.MirroredStrategy()
  # with mirrored_strategy.scope():
  model = _build_keras_model()

  steps_per_epoch = _TRAIN_DATA_SIZE / _TRAIN_BATCH_SIZE

  model.fit(
      train_dataset,
      epochs=int(fn_args.train_steps / steps_per_epoch),
      steps_per_epoch=steps_per_epoch,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }
