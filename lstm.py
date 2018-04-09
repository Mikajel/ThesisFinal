import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import config as cfg
from os import path, getcwd, listdir
import pickle
from sklearn.metrics import roc_auc_score


def load_vector_file(filepath: str) -> ([[float or int]], [float or int]):

    with open(path.join(getcwd(), cfg.dir_vectors, filepath), 'rb') as vector_file:

        vector_tuples = pickle.load(vector_file)

    x = np.array([np.array(vector_tuple[0]) for vector_tuple in vector_tuples])
    y = np.array([vector_tuple[1] for vector_tuple in vector_tuples])

    return x, y


def load_multiple_vector_files(filepaths: [str]):
    """
    Loads multiple vector files and appends vectors together.
    Good idea for test and valid set.
    Not a good idea for train, too much data in RAM, do in batches.
    :param filepaths:
    list of file paths to load vectors from
    :return:
    """

    inputs, targets = load_vector_file(filepaths[0])

    for filepath in filepaths[1:]:

        x, y = load_vector_file(filepath)

        np.concatenate([inputs, x])
        np.concatenate([targets, y])

    return inputs, targets


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


def lstm(x, weight, bias, n_steps, n_classes):

    print(cfg.n_hidden_layers * [cfg.n_hidden_cells_in_layer])

    rnn_layers = [
        tf.nn.rnn_cell.LSTMCell(size, state_is_tuple=True)
        for size in cfg.n_hidden_layers * [cfg.n_hidden_cells_in_layer]
    ]

    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    output, state = tf.nn.dynamic_rnn(multi_rnn_cell, x, dtype=tf.float32)

    # turn output around axis to be 1-dimensional
    output_flattened = tf.reshape(output, [-1, cfg.n_hidden_cells_in_layer])
    output_logits = tf.add(tf.matmul(output_flattened, weight), bias)

    output_all = tf.nn.sigmoid(output_logits)
    output_reshaped = tf.reshape(output_all, [-1, n_steps, n_classes])

    # ??? switch batch size with sequence size. ???
    # then gather last time step values
    output_last = tf.gather(tf.transpose(output_reshaped, [1, 0, 2]), n_steps - 1)

    # output = tf.transpose(output, [1, 0, 2])
    # last = tf.gather(output, int(output.get_shape()[0]) - 1)
    # output_last = tf.nn.sigmoid(tf.matmul(last, weight) + bias)

    return output_last, output_all


def run_thesis():

    # TODO: LIMITING AMOUNT OF FILES
    input_batch_filepaths = sorted(
        [path.join(getcwd(), cfg.dir_vectors, filename)
         for filename in listdir(path.join(getcwd(), cfg.dir_vectors))
         if len(filename.split('_')[2]) < 3
         ]
    )

    # # cutting the last incomplete file(not full wrap amount)
    # input_batch_filepaths = input_batch_filepaths[:-1]
    # print(f'Omitting {input_batch_filepaths[-1]}')

    batch_amount = len(input_batch_filepaths)

    train_amount = int(batch_amount * 0.7)
    test_amount  = int(batch_amount * 0.2)
    valid_amount = int(batch_amount * 0.1)

    train_filepaths = input_batch_filepaths[:train_amount]
    test_filepaths = input_batch_filepaths[train_amount:train_amount+test_amount]
    valid_filepaths = input_batch_filepaths[train_amount+test_amount:]

    print(f'Training:   {cfg.pickle_wrap_amount * len(train_filepaths)} vectors')
    print(f'Testing:    {cfg.pickle_wrap_amount * len(test_filepaths)} vectors')
    print(f'Validation: {cfg.pickle_wrap_amount * len(valid_filepaths)} vectors')

    x_test, y_test = load_multiple_vector_files(test_filepaths)
    x_valid, y_valid = load_multiple_vector_files(valid_filepaths)

    # print(y_test.shape)
    # for test in y_test:
    #     positions = [index for index, value in enumerate(test) if value]
    #     print(f'{sum(test)}: {len(positions)}')
    # return

    n_input, n_steps, n_classes = get_input_target_lengths(check_print=False)

    print(n_steps)
    print(n_input)
    print(n_classes)

    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])
    y_steps = tf.placeholder("float", [None, n_classes])

    weight = weight_variable([cfg.n_hidden_cells_in_layer, n_classes])
    bias = bias_variable([n_classes])

    y_last, y_all = lstm(x, weight, bias, n_steps, n_classes)

    #all_steps_cost=tf.reduce_mean(-tf.reduce_mean((y_steps * tf.log(y_all))+(1 - y_steps) * tf.log(1 - y_all),reduction_indices=1))
    all_steps_cost = -tf.reduce_mean((y_steps * tf.log(y_all)) + (1 - y_steps) * tf.log(1 - y_all))
    last_step_cost = -tf.reduce_mean((y * tf.log(y_last)) + ((1 - y) * tf.log(1 - y_last)))
    loss_function = (cfg.alpha * all_steps_cost) + ((1 - cfg.alpha) * last_step_cost)

    optimizer = tf.train.AdamOptimizer(learning_rate=cfg.learning_rate).minimize(loss_function)

    with tf.Session() as session:

        tf.global_variables_initializer().run()

        for epoch in range(cfg.epoch_amount):
            print(f'\nEpoch number {epoch+1}\n')
            for index, batch_filepath in enumerate(train_filepaths[:5]):

                print(f'Batch number {index+1}')

                x_batch, y_batch = load_vector_file(batch_filepath)
                batch_y_steps = np.tile(y_batch, ((x_batch.shape[1]), 1))

                _, c = session.run(
                    [optimizer, loss_function],
                    feed_dict={
                        x: x_batch,
                        y: y_batch,
                        y_steps: batch_y_steps
                    }
                )

            print('\nTesting after epoch: ')

            y_predictions = session.run(y_last, feed_dict={x: x_test})
            print("ROC AUC Score: ", roc_auc_score(y_test, y_predictions))

        print('Validating: ')
        y_predictions = session.run(y_last, feed_dict={x: x_valid})
        print("ROC AUC Score: ", roc_auc_score(y_valid, y_predictions))
