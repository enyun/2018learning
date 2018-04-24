import tensorflow as tf
from tensorflow.python.estimator.inputs import numpy_io
import numpy as np
import collections
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow import feature_column

from tensorflow.python.feature_column.feature_column import _LazyBuilder


def test_embedding():
    color_data = {'color': [['R'], ['G'], ['B'], ['A']]}  # 4行样本

    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )

    color_embeding = feature_column.embedding_column(color_column, 8)
    color_embeding_dense_tensor = feature_column.input_layer(color_data, [color_embeding])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        session.run(tf.tables_initializer())

        print('embeding' + '_' * 40)
        print(session.run([color_embeding_dense_tensor]))

#test_embedding()

def emb():
    xb = { 'x' : [['a','b'], ['a', 'c'], ['b', 'c']]}
    x = { 'x' : [['a', 'b'], [ 'b', 'c'], ['c', ''], ['', '']]} #可以是多值变量, 对于一篇文章而言是好的处理方式
    fx = feature_column.categorical_column_with_vocabulary_list('x',
                                                                ['a', 'b','c', 'd'],
                                                                dtype=tf.string,
                                                                default_value=0 )
    fex = feature_column.embedding_column( fx, 4, 'mean' )

    t = feature_column.input_layer(x, [fex])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        print(sess.run(t))

emb();