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
from bert_tokenizer_utils import Special_Bert_Tokenizer
from bert_models import BertForSingleSentenceSentimentAnalysis

from tfx.components.trainer.executor import TrainerFnArgs

_TRAIN_BATCH_SIZE = 64
_EVAL_BATCH_SIZE = 64
_LABEL_KEY = "label"
_BERT_LINK = "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/2"
_MAX_LEN = 128
_EPOCHS = 1

def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(
          filenames,
          compression_type='GZIP')

def _tokenize(feature):
    """Tokenize the two sentences and insert appropriate tokens"""
    vocab_dir = '/home/tommywei/bert_cola/assets/vocab.txt'
    tokenizer = Special_Bert_Tokenizer(vocab_dir)
    return tokenizer.tokenize_single_sentence(
      tf.reshape(feature, [-1]),
      max_len=_MAX_LEN)

def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  feature = inputs['feature']
  label = inputs['label']
  input_word_ids, input_mask, segment_ids = _tokenize(feature)
  return {
          'label': label,
          'input_word_ids': input_word_ids,
          'input_mask': input_mask,
          'segment_ids': segment_ids
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
  
def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""

  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = model.tft_layer(parsed_features)
    # TODO(b/148082271): Remove this line once TFT 0.22 is used.
    transformed_features.pop(_transformed_name(_LABEL_KEY), None)

    return model(transformed_features)

  return serve_tf_examples_fn

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
  bert_layer = hub.KerasLayer(_BERT_LINK, trainable=True)
  model = BertForSingleSentenceSentimentAnalysis(bert_layer, _MAX_LEN)

  model.fit(
      train_dataset,
      epochs=_EPOCHS,
      steps_per_epoch=fn_args.train_steps,
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
