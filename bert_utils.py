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

import os
import absl
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft
import tensorflow_hub as hub
import tensorflow_text as text

from tfx.components.trainer.executor import TrainerFnArgs

_TRAIN_BATCH_SIZE = 512
_TRAIN_DATA_SIZE = 51200
_EVAL_BATCH_SIZE = 512
_LABEL_KEY = "sentiment"

def regex_split_with_offsets(input,
                             delim_regex_pattern,
                             keep_delim_regex_pattern="",
                             name=None):
  r"""Split `input` by delimiters that match a regex pattern; returns offsets.
  `regex_split_with_offsets` will split `input` using delimiters that match a
  regex pattern in `delim_regex_pattern`. Here is an example:
  ```
  text_input=["hello there"]
  # split by whitespace
  result, begin, end = regex_split_with_offsets(text_input, "\s")
  # result = [["hello", "there"]]
  # begin = [[0, 7]]
  # end = [[5, 11]]
  ```
  By default, delimiters are not included in the split string results.
  Delimiters may be included by specifying a regex pattern
  `keep_delim_regex_pattern`. For example:
  ```
  text_input=["hello there"]
  # split by whitespace
  result, begin, end = regex_split_with_offsets(text_input, "\s", "\s")
  # result = [["hello", " ", "there"]]
  # begin = [[0, 5, 7]]
  # end = [[5, 6, 11]]
  ```
  If there are multiple delimiters in a row, there are no empty splits emitted.
  For example:
  ```
  text_input=["hello  there"]  # two continuous whitespace characters
  # split by whitespace
  result, begin, end = regex_split_with_offsets(text_input, "\s")
  # result = [["hello", "there"]]
  ```
  See https://github.com/google/re2/wiki/Syntax for the full list of supported
  expressions.
  Args:
    input: A Tensor or RaggedTensor of string input.
    delim_regex_pattern: A string containing the regex pattern of a delimiter.
    keep_delim_regex_pattern: (optional) Regex pattern of delimiters that should
      be kept in the result.
    name: (optional) Name of the op.
  Returns:
    A tuple of RaggedTensors containing:
      (split_results, begin_offsets, end_offsets)
    where tokens is of type string, begin_offsets and end_offsets are of type
    int64.
  """
  delim_regex_pattern = b"".join(
      [b"(", delim_regex_pattern.encode("utf-8"), b")"])
  keep_delim_regex_pattern = b"".join(
      [b"(", keep_delim_regex_pattern.encode("utf-8"), b")"])

  # Convert input to ragged or tensor
  input = ragged_tensor.convert_to_tensor_or_ragged_tensor(
      input, dtype=dtypes.string)

  if ragged_tensor.is_ragged(input):
    # send flat_values to regex_split op.
    tokens, begin_offsets, end_offsets, row_splits = (
        gen_regex_split_ops.regex_split_with_offsets(
            input.flat_values,
            delim_regex_pattern,
            keep_delim_regex_pattern,
            name=name))

    # Pack back into original ragged tensor
    tokens_rt = ragged_tensor.RaggedTensor.from_row_splits(tokens, row_splits)
    tokens_rt = ragged_tensor.RaggedTensor.from_row_splits(
        tokens_rt, input.row_splits)
    begin_offsets_rt = ragged_tensor.RaggedTensor.from_row_splits(
        begin_offsets, row_splits)
    begin_offsets_rt = ragged_tensor.RaggedTensor.from_row_splits(
        begin_offsets_rt, input.row_splits)
    end_offsets_rt = ragged_tensor.RaggedTensor.from_row_splits(
        end_offsets, row_splits)
    end_offsets_rt = ragged_tensor.RaggedTensor.from_row_splits(
        end_offsets_rt, input.row_splits)
    return tokens_rt, begin_offsets_rt, end_offsets_rt

  else:
    # send flat_values to regex_split op.
    tokens, begin_offsets, end_offsets, row_splits = (
        gen_regex_split_ops.regex_split_with_offsets(input, delim_regex_pattern,
                                                     keep_delim_regex_pattern))

    # Pack back into ragged tensors
    tokens_rt = ragged_tensor.RaggedTensor.from_row_splits(
        tokens, row_splits=row_splits)
    begin_offsets_rt = ragged_tensor.RaggedTensor.from_row_splits(
        begin_offsets, row_splits=row_splits)
    end_offsets_rt = ragged_tensor.RaggedTensor.from_row_splits(
        end_offsets, row_splits=row_splits)
    return tokens_rt, begin_offsets_rt, end_offsets_rt


# pylint: disable= redefined-builtin
def regex_split(input,
                delim_regex_pattern,
                keep_delim_regex_pattern="",
                name=None):
  r"""Split `input` by delimiters that match a regex pattern.
  `regex_split` will split `input` using delimiters that match a
  regex pattern in `delim_regex_pattern`. Here is an example:
  ```
  text_input=["hello there"]
  # split by whitespace
  result, begin, end = regex_split_with_offsets(text_input, "\s")
  # result = [["hello", "there"]]
  ```
  By default, delimiters are not included in the split string results.
  Delimiters may be included by specifying a regex pattern
  `keep_delim_regex_pattern`. For example:
  ```
  text_input=["hello there"]
  # split by whitespace
  result, begin, end = regex_split_with_offsets(text_input, "\s", "\s")
  # result = [["hello", " ", "there"]]
  ```
  If there are multiple delimiters in a row, there are no empty splits emitted.
  For example:
  ```
  text_input=["hello  there"]  # two continuous whitespace characters
  # split by whitespace
  result, begin, end = regex_split_with_offsets(text_input, "\s")
  # result = [["hello", "there"]]
  ```
  See https://github.com/google/re2/wiki/Syntax for the full list of supported
  expressions.
  Args:
    input: A Tensor or RaggedTensor of string input.
    delim_regex_pattern: A string containing the regex pattern of a delimiter.
    keep_delim_regex_pattern: (optional) Regex pattern of delimiters that should
      be kept in the result.
    name: (optional) Name of the op.
  Returns:
    A RaggedTensors containing of type string containing the split string
    pieces.
  """
  tokens, _, _ = regex_split_with_offsets(input, delim_regex_pattern,
                                          keep_delim_regex_pattern, name)
  return tokens

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
    tokenizer = text.BertTokenizer(
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
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  stringA = inputs['stringA']
  stringB = inputs['stringB']
  label = inputs['Quality']
  return {
          'label': label,
          'stringA': _tokenize(stringA, stringB),
          'stringB': stringB
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
