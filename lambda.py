import ctypes
import os
from six.moves import urllib
import zipfile
import stat
import logging
#import shutil

logging.basicConfig(level=logging.DEBUG)
print "logging"

#if os.path.exists("/tmp/.theano"):
#    shutil.rmtree("/tmp/.theano")



def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
       results[i, sequence] = 1. 
    return results

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


model = keras.models.load_model('my_sms_model.h5')


def handler(event, context):

    sms = event['sentence']
    print sms
    print type(sms)

    sentence = one_hot(str(sms), 87413)
    print sentence
    sentences = [sentence]

    vector = vectorize_sequences(sentences,87413)

    result = model.predict(vector)  
    print "Verdict: ", result
    print "Shape: ", result.shape
    print "Type: ", type(result)
    result = float(np.array2string(result)[2:-2])
    print(result)   
    return result

