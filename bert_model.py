import tensorflow as tf
import tf.keras as keras
import tf.hub as hub

_BERT_BASE = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
_BERT_LARGE = ""
def build_bert_for_classification():     
