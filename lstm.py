import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import config as cfg
from os import path, getcwd, listdir
import pickle

input_batch_filepaths = [path.join(getcwd(), cfg.dir_vectors, filename)
                         for filename in listdir(path.join(getcwd(), cfg.dir_users))]

batch_amount = len(input_batch_filepaths)

train_amount = int(batch_amount*0.6)
test_amount = int(batch_amount*0.3)
valid_amount = int(batch_amount*0.1)

train_filepaths = input_batch_filepaths[:train_amount]
test_filepaths = input_batch_filepaths[train_amount:train_amount+test_amount]
valid_filepaths = input_batch_filepaths[train_amount+test_amount:]

print(len(train_filepaths))
print(len(test_filepaths))
print(len(valid_filepaths))


def get_input_target_lengths(check_print: bool=False) -> (int, int, int):
    """
    Get vector shapes
    :param check_print:
    bool, print shapes into console
    :return:
    size of input vector, number of steps, size of output vector
    """

    with open(path.join(getcwd(), cfg.dir_vectors, cfg.vectors_baseline_filename + '3'), 'rb') as vector_file:

        all_users_vectors = pickle.load(vector_file)

    sample_user_tuple = all_users_vectors[0]

    if check_print:
        print(f'Amount of sample users:\n{len(all_users_vectors)}')
        print(f'Amount of input time events:\n{len(all_users_vectors[0][0])}\n\n')

        for pair in all_users_vectors:
            print(f'Length of single input vector: {len(pair[0][0])}')
            print(f'Length of single target vector:{len(pair[1])}')

        print('Sample user tuple: ')
        print('Inputs:')
        print(sample_user_tuple[0])
        print('Target:')
        print(sample_user_tuple[1])
        print('Ones in sample target:')
        print(f'{sum(sample_user_tuple[1])}')

    return len(sample_user_tuple[0][0]), len(all_users_vectors[0][0]), len(sample_user_tuple[1])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def lstm(x, weight, bias, input_length, n_steps, n_classes):
    cell = rnn_cell.LSTMCell(cfg.n_hidden_cells_in_layer, state_is_tuple=True)
    multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * cfg.n_hidden_layers)

    output, state = tf.nn.dynamic_rnn(multi_layer_cell, x, dtype=tf.float32)

    output_flattened = tf.reshape(output, [-1, cfg.n_hidden_cells_in_layer])
    output_logits = tf.add(tf.matmul(output_flattened, weight), bias)

    output_all = tf.nn.sigmoid(output_logits)
    output_reshaped = tf.reshape(output_all, [-1, n_steps, n_classes])
    output_last = tf.gather(tf.transpose(output_reshaped, [1, 0, 2]), n_steps - 1)

    # output = tf.transpose(output, [1, 0, 2])
    # last = tf.gather(output, int(output.get_shape()[0]) - 1)
    # output_last = tf.nn.sigmoid(tf.matmul(last, weight) + bias)

    return output_last, output_all


n_input, n_steps, n_classes = get_input_target_lengths(check_print=True)

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
y_steps = tf.placeholder("float", [None, n_classes])







"""
# one file == one batch
batch_size = cfg.pickle_wrap_amount

data = tf.placeholder(tf.float32, [batch_size, None, length_input])
target = tf.placeholder(tf.float32, [batch_size, None, length_output])

# FIXME: using single layer for now
# lstm_stacked = tf.contrib.rnn.MultiRNNCell(
#     [lstm_cell(hidden_cells=cfg.n_hidden_cells_in_layer) for _ in range(cfg.n_hidden_layers)]
# )
cell = lstm_cell(hidden_cells=cfg.n_hidden_cells_in_layer)

outputs, final_state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)


val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = weight_variable(target)
bias = bias_variable(target)
prediction = tf.nn.sigmoid(tf.matmul(last, weight) + bias)

cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))

optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)



init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)


# batch loop
for filepath in input_batch_filepaths:
    with open(filepath, 'rb') as batch_file:

        # set of users from file == batch
        batch_users = pickle.load(batch_file)
        for user in batch_users:

            inputs  = [vector_tuple[0] for vector_tuple in batch_users]
            targets = [vector_tuple[1] for vector_tuple in batch_users]


"""



# def LSTM(x, weight, bias):
#
#     cell = rnn_cell.LSTMCell(n_hidden, state_is_tuple=True)
#     multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)
#     output, state = tf.nn.dynamic_rnn(multi_layer_cell, x, dtype=tf.float32)
#     output_flattened = tf.reshape(output, [-1, n_hidden])
#     output_logits = tf.add(tf.matmul(output_flattened, weight), bias)
#     output_all = tf.nn.sigmoid(output_logits)
#     output_reshaped = tf.reshape(output_all, [-1, n_steps, n_classes])
#     output_last = tf.gather(tf.transpose(output_reshaped, [1, 0, 2]), n_steps - 1)
#     #output = tf.transpose(output, [1, 0, 2])
#     #last = tf.gather(output, int(output.get_shape()[0]) - 1)
#     #output_last = tf.nn.sigmoid(tf.matmul(last, weight) + bias)
#     return output_last, output_all