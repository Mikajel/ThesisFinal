import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import config as cfg
from os import path, getcwd, listdir, mkdir
import pickle
from sklearn.metrics import roc_auc_score, f1_score
from random import shuffle
from collections import OrderedDict
from imblearn.over_sampling import SMOTE
import time
import datetime


def load_shuffled_train_valid_test(
        filepaths: [str],
        ratios: tuple,
        required_sample_amount: int,
        flag_oversample: bool,
        flag_undersample: bool,
        sampling_target_amount: int):

    all_user_vectors = []

    partners_list = []
    partner_numerosities = {}

    for filepath in filepaths:

        with open(filepath, 'rb') as vector_file:

            current_file_vectors = pickle.load(vector_file)

            if len(current_file_vectors) >= required_sample_amount:

                if flag_undersample:
                    # undersample
                    if len(current_file_vectors) > sampling_target_amount:
                        current_file_vectors = current_file_vectors[:sampling_target_amount]

                all_user_vectors += current_file_vectors

            partner_id = int(filepath.split('_')[-1])
            partners_list.append(partner_id)
            partner_vector_amount = len(current_file_vectors)
            partner_numerosities[partner_id] = partner_vector_amount

    # SMOTE
    if flag_oversample:
        all_vector_inputs = [vector_pair[0] for vector_pair in all_user_vectors]
        all_vector_targets = [vector_pair[1] for vector_pair in all_user_vectors]

        inputs_res, targets_res = SMOTE(kind='svm').fit_sample(all_vector_inputs, all_vector_targets)

        all_user_vectors = [(inputs_res[index], targets_res[index]) for index in range(len(inputs_res))]

    # SMOTE

    max_vector_amount = max(partner_numerosities.values())
    partners_list.sort()

    class_weights = []

    for partner_id in partners_list:

        current_class_weight = (1 - partner_numerosities[partner_id] / (max_vector_amount * 2)) ** 4
        class_weights.append(
            current_class_weight
        )

        print('{} : {} : {:.4f}'.format(partner_id, partner_numerosities[partner_id], current_class_weight))

    shuffle(all_user_vectors)

    train_ratio, valid_ratio, test_ratio = ratios
    n_vectors = len(all_user_vectors)

    train = all_user_vectors[:int(n_vectors * train_ratio)]
    valid = all_user_vectors[int(n_vectors * train_ratio):int(n_vectors * (train_ratio + valid_ratio))]
    test = all_user_vectors[int(n_vectors * (train_ratio + valid_ratio)):]

    print('Train vectors: {}'.format(len(train)))
    print('Valid vectors: {}'.format(len(valid)))
    print('Test  vectors: {}'.format(len(test)))

    x_train = np.array([np.array(vector[0]) for vector in train])
    y_train = np.array([np.array(vector[1]) for vector in train])

    x_valid = np.array([np.array(vector[0]) for vector in valid])
    y_valid = np.array([np.array(vector[1]) for vector in valid])

    x_test = np.array([np.array(vector[0]) for vector in test])
    y_test = np.array([np.array(vector[1]) for vector in test])

    return x_train, y_train, x_valid, y_valid, x_test, y_test, class_weights


def get_input_target_lengths(flag_heavy_vectors: bool, check_print: bool = False) -> (int, int, int):
    """
    Get vector shapes
    :param check_print:
    bool, print shapes into console
    :param flag_heavy_vectors:
    use vectors with partner_id one-hot subvector on inputs
    :return:
    size of input vector, number of steps, size of output vector
    """

    if flag_heavy_vectors:
        sample_filepath = path.join(
            getcwd(),
            cfg.dir_sorted_partner_vectors_input_with_partners,
            cfg.sorted_heavy_partner_vectors_baseline_filename + '13295'
        )
    else:
        sample_filepath = path.join(
            getcwd(),
            cfg.dir_partners_grouped_vectors,
            cfg.sorted_partner_vectors_baseline_filename + '13295'
        )

    with open(sample_filepath, 'rb') as vector_file:

        all_users_vectors = pickle.load(vector_file)

    sample_user_tuple = all_users_vectors[0]

    if check_print:
        print('Amount of sample users:\n{}'.format(len(all_users_vectors)))
        print('Amount of input time events:\n{}\n\n'.format(len(all_users_vectors[0][0])))

        for pair in all_users_vectors:
            print('Length of single input vector: {}'.format(len(pair[0][0])))
            print('Length of single target vector:{}'.format(len(pair[1])))

        print('Sample user tuple: ')
        print('Inputs:')
        print(sample_user_tuple[0])
        print('Target:')
        print(sample_user_tuple[1])
        print('Ones in sample target:')
        print('{}'.format(sum(sample_user_tuple[1])))

    return len(sample_user_tuple[0][0]), len(all_users_vectors[0][0]), len(sample_user_tuple[1])


def score(logits, targets, weights) -> float:

    batch_score = 0.0

    for _ in range(len(logits)):

        pass

    return batch_score



def RNN_1(
        x: tf.placeholder,
        n_input: int,
        n_steps: int,
        n_classes: int,
        n_hidden_cells: int,
        dropout_chance: float,
        model_type: str,
        train: bool):

    def get_weights(layer_from_size: int, layer_to_size: int):

        return tf.Variable(tf.random_normal([layer_from_size, layer_to_size]))

    def get_biases(layer_to_size: int):

        return tf.Variable(tf.random_normal([layer_to_size]))

    x = tf.unstack(x, n_steps, 1)

    if model_type == 'rnn':
        cell_constructor = rnn.BasicRNNCell
    elif model_type == 'gru':
        cell_constructor = rnn.GRUCell
    elif model_type == 'lstm':
        cell_constructor = rnn.BasicLSTMCell
    elif model_type == 'nas':
        cell_constructor = rnn.NASCell
    else:
        raise Exception("model type not supported: {}".format(model_type))

    cell = cell_constructor(n_hidden_cells, activation=tf.nn.relu)
    if train:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1 - dropout_chance)

    outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)

    weights_out = get_weights(n_hidden_cells, n_classes)
    biases_out = get_biases(n_classes)

    output_layer = tf.matmul(outputs[-1], weights_out) + biases_out

    return output_layer


def load_dataset(flag_heavy_vectors: bool, flag_undersample: bool, flag_oversample: bool):

    dataset_split_ratio = cfg.dataset_split
    sampling_target_amount = 100
    required_sample_amount = 50

    if flag_heavy_vectors:
        input_batch_filepaths = sorted(
            [path.join(getcwd(), cfg.dir_sorted_partner_vectors_input_with_partners, filename)
             for filename in listdir(path.join(getcwd(), cfg.dir_sorted_partner_vectors_input_with_partners))]
        )
    else:
        input_batch_filepaths = sorted(
            [path.join(getcwd(), cfg.dir_partners_grouped_vectors, filename)
             for filename in listdir(path.join(getcwd(), cfg.dir_partners_grouped_vectors))]
        )

    print('Loading dataset into train, valid, test')
    dataset_loading_start = time.time()

    x_train, y_train, x_valid, y_valid, x_test, y_test, class_weights = load_shuffled_train_valid_test(
        filepaths=input_batch_filepaths,
        ratios=dataset_split_ratio,
        required_sample_amount=required_sample_amount,
        flag_oversample=flag_oversample,
        flag_undersample=flag_undersample,
        sampling_target_amount=sampling_target_amount
    )

    dataset_loading_end = time.time()
    print('Loading finished in {}\n'.format(dataset_loading_end - dataset_loading_start))

    return x_train, y_train, x_valid, y_valid, x_test, y_test, class_weights


def run_training(
        *,
        dataset: tuple,
        n_hidden_cells: int,
        learning_rate: float,
        alpha: float,
        min_epoch_amount: int,
        batch_size: int,
        dropout_chance: float,
        model_type: str,
        optimizer_type: str,
        use_class_weights: bool,
        train: bool,
        flag_heavy_vectors: bool):

    logtime_begin = str(datetime.datetime.now())

    log = 'Logging start: {}\n\n'.format(logtime_begin)

    log += 'Dataset hyperparameters:\n'
    log += 'Using heavy vectors:  {}\n'.format(cfg.flag_heavy_vectors)
    log += '\tDataset split:\n'
    log += '\t\tTrain: {}\n'.format(cfg.dataset_split[0])
    log += '\t\tTest:  {}\n'.format(cfg.dataset_split[1])
    log += '\t\tValid: {}\n'.format(cfg.dataset_split[2])
    log += '\tUndersample to:     {}\n'.format(cfg.target_sample_amount)
    log += '\tMin class samples:  {}\n\n'.format(cfg.required_sample_amount)

    log += 'Network hyperparameters:\n'
    log += '\tCell:               {}\n'.format(model_type)
    log += '\tOptimizer:          {}\n'.format(optimizer_type)
    log += '\tWeights:            {}\n'.format(use_class_weights)
    log += '\tLearning rate:      {}\n'.format(learning_rate)
    log += '\tBatch size:         {}\n'.format(batch_size)
    log += '\tDropout:            {}\n'.format(dropout_chance)

    x_train, y_train, x_valid, y_valid, x_test, y_test, class_weights = dataset

    n_input, n_steps, n_classes = get_input_target_lengths(flag_heavy_vectors, check_print=False)

    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])

    logits = RNN_1(
        x=x,
        n_input=n_input,
        n_steps=n_steps,
        n_classes=n_classes,
        n_hidden_cells=n_hidden_cells,
        dropout_chance=dropout_chance,
        model_type=model_type,
        train=train
    )

    if optimizer_type == 'adam':
        optimizer_constructor = tf.train.AdamOptimizer
    elif optimizer_type == 'sgd':
        optimizer_constructor = tf.train.GradientDescentOptimizer
    elif optimizer_type == 'rmsp':
        optimizer_constructor = tf.train.RMSPropOptimizer
    elif optimizer_type == 'adg':
        optimizer_constructor = tf.train.AdagradOptimizer
    else:
        raise ValueError('Value {} is not a valid string for declaring optimizer type'.format(optimizer_type))

    predictions = tf.nn.softmax(logits)

    tensor_class_weights = tf.constant(class_weights)
    tensor_class_weights = tf.reshape(tensor_class_weights, [1, n_classes])
    matrix_class_weights = tf.tile(
                    tensor_class_weights,
                    [batch_size, 1]
                )

    correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    rewards = tf.reduce_sum(tf.multiply(matrix_class_weights, y), 1)

    score = tf.reduce_sum(
        tf.multiply(
            tf.cast(correct_pred, tf.float32),
            rewards
        )
    )

    max_score = tf.reduce_sum(
        tf.multiply(
            matrix_class_weights,
            y
        )
    )

    if use_class_weights:
        loss_function = tf.reduce_mean(
            tf.multiply(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits,
                    labels=y
                ),
                rewards
            )
        )
    else:
        loss_function = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=y
            )
        )

    optimizer = optimizer_constructor(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_function)

    init = tf.global_variables_initializer()

    with tf.Session() as session:

        session.run(init)

        print('///////////////////////////////////////////////////////////////////////////////////')

        epoch = 0
        epoch_valid_accuracies = []
        stop_training = False

        while (not stop_training) or (epoch < min_epoch_amount):

            epoch += 1

            log += '\nEpoch number {}\n'.format(epoch)

            batch_train_accuracies = []

            x_train_shuffled = []
            y_train_shuffled = []

            shuffled_tuples = list(zip(x_train, y_train))
            shuffle(shuffled_tuples)

            [x_train_shuffled.append(shuffled_pair[0]) for shuffled_pair in shuffled_tuples]
            [y_train_shuffled.append(shuffled_pair[1]) for shuffled_pair in shuffled_tuples]

            for index in range(0, len(x_train_shuffled), batch_size):
                x_batch_train = x_train_shuffled[index:index + batch_size]
                y_batch_train = y_train_shuffled[index:index + batch_size]

                try:
                    session.run(train_op, feed_dict={x: x_batch_train, y: y_batch_train})

                    loss, batch_accuracy = session.run(
                        [loss_function, accuracy],
                        feed_dict={
                            x: x_batch_train,
                            y: y_batch_train
                        }
                    )

                    batch_train_accuracies.append(batch_accuracy)
                # error with the last batch having different shape
                except tf.errors.InvalidArgumentError:
                    pass

            log += '\nValidating after epoch: \n'

            batch_valid_accuracies = []
            valid_score = 0
            max_valid_score = 0

            for index in range(0, len(x_valid), batch_size):
                x_batch_valid = x_valid[index:index + batch_size]
                y_batch_valid = y_valid[index:index + batch_size]

                try:
                    bva, bvs, ts = session.run([accuracy, score, max_score], feed_dict={x: x_batch_valid, y: y_batch_valid})

                    batch_valid_accuracies.append(bva)
                    valid_score += bvs
                    max_valid_score += ts

                # for the last batch, incompatible shape
                except tf.errors.InvalidArgumentError:
                    pass

            epoch_train_accuracy = 100 * np.mean(batch_train_accuracies)
            epoch_valid_accuracy = 100 * np.mean(batch_valid_accuracies)

            log += 'Loss after epoch:    {0:.4f}\n'.format(loss)
            log += 'Average train batch accuracy: {0:.4f}%\n'.format(epoch_train_accuracy)
            log += 'Validation accuracy: {0:.4f}%\n'.format(epoch_valid_accuracy)
            log += 'Validation score: {0:.4f}\n\n'.format(valid_score)

            print('Epoch:     {}'.format(epoch))
            print('Score:     {0:.4f}'.format(valid_score))
            print('% score:   {0:.4f}\n'.format(100*valid_score/max_valid_score))

            print('Loss: {:.4f}\n'.format(loss))

            print('Avg train batch accuracy: {0:.4f}%'.format(epoch_train_accuracy))
            print('Validation accuracy:      {0:.4f}%\n'.format(epoch_valid_accuracy))

            if (np.mean(epoch_valid_accuracies[-10:]) > epoch_valid_accuracy) and (epoch >= min_epoch_amount):
                log += 'Stopping training because:\n'
                log += '\tAverage valid accuracy(current epoch): {0:.4f}%\n'.format(epoch_valid_accuracy)
                log += '\tAverage valid accuracy(last N epochs): {0:.4f}%\n'.format(np.mean(epoch_valid_accuracies[-10:]))

                stop_training = True

            epoch_valid_accuracies.append(epoch_valid_accuracy)

        log += '\n\nFinal testing after all epochs: \n'

        batch_test_accuracies = []
        test_score = 0
        max_test_score = 0

        for index in range(0, len(x_test), batch_size):
            x_batch_test = x_test[index:index + batch_size]
            y_batch_test = y_test[index:index + batch_size]

            try:
                bta, bts, ts = session.run([accuracy, score, max_score], feed_dict={x: x_batch_test, y: y_batch_test})
                batch_test_accuracies.append(bta)
                test_score += bts
                max_test_score += ts

            # for the last batch, incompatible shape
            except tf.errors.InvalidArgumentError:
                pass

        print('Epoch:     {}'.format(epoch))
        print('Score:     {}'.format(test_score))
        print('Max score: {}'.format(max_test_score))
        print('% score:   {0:.4f}\n'.format(100 * test_score / max_test_score))

        log += 'Test accuracy: {0:.4f}%\n'.format(100 * np.mean(batch_test_accuracies))
        log += 'Test score: {0:.4f}%\n\n'.format(test_score)

        dir_name = str(logtime_begin).split(' ')[0]

        if path.exists(path.join(getcwd(), cfg.dir_logging, dir_name)):
            pass

        else:
            mkdir(path.join(getcwd(), cfg.dir_logging, dir_name))

        with open(path.join(getcwd(), cfg.dir_logging, dir_name, logtime_begin), 'w') as logfile:

            logfile.write(log)
