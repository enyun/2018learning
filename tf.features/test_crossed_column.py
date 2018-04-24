import tensorflow as tf
from tensorflow.python.estimator.inputs import numpy_io
import numpy as np
import collections
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow import feature_column

from tensorflow.python.feature_column.feature_column import _LazyBuilder

def test_crossed_column():
    featrues = {
        'price': [['A', 'A'], ['B', 'D'], ['C', 'A']],
        'color': [['R', 'R'], ['G', 'G'], ['B', 'B']]
    }

    price = feature_column.categorical_column_with_vocabulary_list('price', ['A', 'B', 'C', 'D'])
    color = feature_column.categorical_column_with_vocabulary_list('color', ['R', 'G', 'B'])
    p_x_c = feature_column.crossed_column([price, color], 16)
    p_x_c_identy = feature_column.indicator_column(p_x_c) # what's this?
    p_x_c_identy_dense_tensor = feature_column.input_layer(featrues, [p_x_c_identy])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run([p_x_c_identy_dense_tensor]))

test_crossed_column()