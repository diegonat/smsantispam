import numpy as np
import os
import tensorflow as tf
import pandas
import json
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_output import PredictOutput

INPUT_TENSOR_NAME = 'inputs'
SIGNATURE_NAME = "serving_default"
LEARNING_RATE = 0.01

def model_fn(features, labels, mode, params):

    first_hidden_layer = tf.keras.layers.Dense(16, activation='relu', name='first-layer')(features[INPUT_TENSOR_NAME])
    second_hidden_layer = tf.keras.layers.Dense(16, activation='relu')(first_hidden_layer)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(second_hidden_layer)
    predictions = output_layer

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"spam": predictions},
            export_outputs={SIGNATURE_NAME: PredictOutput({"spam": predictions})})

    # 2. Define the loss function for training/evaluation using Tensorflow.
    loss = tf.losses.log_loss(labels, predictions)

    # 3. Define the training operation/optimizer using Tensorflow operation/optimizer.
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=LEARNING_RATE,
        optimizer="SGD")

    # 4. Generate predictions as Tensorflow tensors.
    predictions_dict = {"spam": predictions}

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels, predictions)
    }

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)

#def keras_model_fn(hyperparameters):
#    model = tf.keras.models.Sequential()
#    model.add(tf.keras.layers.Dense(16, activation='relu',input_shape=(9013,), name='inputs'))
#    model.add(tf.keras.layers.Dense(16, activation='relu'))
#    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#    model.compile(loss='binary_crossentropy',
#                  optimizer='rmsprop',
#                  metrics=['accuracy'])
#    return model

def train_input_fn(training_dir, params):
    return _input_fn(training_dir, 'sms_train_set.gz')


def eval_input_fn(training_dir, params):
    return _input_fn(training_dir, 'sms_test_set.gz')


def _input_fn(training_dir, training_filename):
    df = pandas.read_csv(os.path.join(training_dir, training_filename))
    
    X = df[df.columns[1:]].values.astype(dtype=np.float32)
    # Reshaping labels as expected by TF
    y = df[df.columns[0]].values.reshape((-1, 1)).astype(dtype=np.float32)
    
    return tf.estimator.inputs.numpy_input_fn(
        x={INPUT_TENSOR_NAME: X},
        y=y,
        num_epochs=None,
        shuffle=True)()

def serving_input_fn(params):
    tensor = tf.placeholder(tf.float32, shape=[1, 9013])
    return build_raw_serving_input_receiver_fn({INPUT_TENSOR_NAME: tensor})()
