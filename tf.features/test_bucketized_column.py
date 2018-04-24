import tensorflow as tf
import numpy as np

import tensorflow as tf
from tensorflow.python.estimator.inputs import numpy_io
import numpy as np
import collections
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow import feature_column

from tensorflow.python.feature_column.feature_column import _LazyBuilder


def test_bucketized_column0():

    price = {'price': [[5.], [15.], [25.], [35.]]}  # 4行样本

    price_column = feature_column.numeric_column('price')
    bucket_price = feature_column.bucketized_column(price_column, [0, 10, 20, 30, 40])

    price_bucket_tensor = feature_column.input_layer(price, [bucket_price])

    with tf.Session() as session:
        print(session.run([price_bucket_tensor]))

def test_bucketized_column():

    price = {'price': [[5. ], [15. ], [25. ], [35. ]]}  # 4行样本

    price_column = feature_column.numeric_column('price')
    bucket_price = feature_column.bucketized_column(price_column, [0, 10, 20, 30, 40])

    price_bucket_tensor = feature_column.input_layer(price, [bucket_price])

    with tf.Session() as session:
        print(session.run([price_bucket_tensor]))


#test_bucketized_column()

def pratise():
    d = {'x': [[32], [16], [38], [98]]}
    cd = feature_column.numeric_column('x')
    bcd = feature_column.bucketized_column(cd, [10, 20, 40, 60])
    fcd = feature_column.input_layer(d, [bcd])

    with tf.Session() as sess:
        print(sess.run(fcd))

pratise()