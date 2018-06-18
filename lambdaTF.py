import ctypes
import os
from six.moves import urllib
import zipfile
import stat
import logging
import json

logging.basicConfig(level=logging.DEBUG)
print "logging"

def response(status_code, response_body):
    return {
                'statusCode': status_code,
                'body': response_body,
                'headers': {
                    'Content-Type': 'application/json',
                },
            }

def vectorize_sequences(sequences, dimension):
    results = numpy.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
       results[i, sequence] = 1. 
    return results

def one_hot_encode(messages, dimension):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_lenght)
        data.append(temp)
    return data

for d, _, files in os.walk('lib'):
    for f in files:
        if f.endswith('.a') or f.endswith('.settings'):
            continue
        print('loading %s...' % f)
        ctypes.cdll.LoadLibrary(os.path.join(d, f))

import keras
from keras.preprocessing.text import one_hot
from numpy import array
import numpy as np
import tensorflow as tf




path_to_model = os.environ['MODEL_PATH']
vocabulary_lenght = 9013



def handler(event, context):
    print event
    #params = event['queryStringParameters']

    sms = event['body']

    print sms

    test_messages = [sms]


    one_hot_test_messages = one_hot_encode(test_messages, vocabulary_lenght)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_lenght)

    with tf.Session(graph=tf.Graph()) as sess:
       tf.saved_model.loader.load(
           sess,
           [tf.saved_model.tag_constants.SERVING],
           path_to_model)


       sigmoid_tensor = sess.graph.get_tensor_by_name('dense_1/Sigmoid:0')
       predictions = sess.run(sigmoid_tensor, {'Placeholder_1:0': encoded_test_messages})

       print(predictions[0][0])

       result = predictions[0][0]

       return response(200, result)


