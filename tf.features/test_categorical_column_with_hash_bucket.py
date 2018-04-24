import tensorflow as tf
from tensorflow.python.estimator.inputs import numpy_io
import numpy as np
import collections
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow import feature_column
from tensorflow.python.feature_column.feature_column import _LazyBuilder

def test_categorical_column_with_hash_bucket0():
    color_data = {'color': [['R'], ['G'], ['B'], ['A']]}  # 4行样本
    builder = _LazyBuilder(color_data)
    color_column = feature_column.categorical_column_with_hash_bucket('color', 7)
    color_column_tensor = color_column._get_sparse_tensors(builder)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        session.run(tf.tables_initializer())

        print(session.run([color_column_tensor.id_tensor]))

    # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
    color_column_identy = feature_column.indicator_column(color_column)

    color_dense_tensor = feature_column.input_layer(color_data, [color_column_identy])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        session.run(tf.tables_initializer())

        print('use input_layer' + '_' * 40)
        print(session.run([color_dense_tensor]))

def test_categorical_column_with_hash_bucket():

    color_data = {'color': [['R', 'G'], ['G', 'A'], ['B', 'G'], ['A', 'G'], ['A', '']]}  # 4行样本
    builder = _LazyBuilder(color_data)

    color_column = feature_column.categorical_column_with_hash_bucket('color', 7)
    color_column_tensor = color_column._get_sparse_tensors(builder).id_tensor

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor]))
        #print(session.run([color_column_tensor.id_tensor]))

    # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
    color_column_identy = feature_column.indicator_column(color_column)

    color_dense_tensor = feature_column.input_layer(color_data, [color_column_identy])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        session.run(tf.tables_initializer())

        print('use input_layer' + '_' * 40)
        print(session.run([color_dense_tensor]))
#test_categorical_column_with_hash_bucket()

def practise():
    fx = {'x' : [['a', 'a'], ['b', 'c'], ['c','e'], ['d', ''], ['e', 'f']]}
    fc = feature_column.categorical_column_with_hash_bucket('x', 5)
    fic = feature_column.indicator_column(fc)
    t2 = fc._get_sparse_tensors(_LazyBuilder(fx)).id_tensor
    tsor = feature_column.input_layer(fx, fic)

    with tf.Session() as sess:
        print(sess.run(t2))
        print(sess.run(tsor))


practise()
