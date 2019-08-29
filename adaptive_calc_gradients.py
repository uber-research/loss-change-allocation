'''
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import time
import h5py
import argparse
import os
import sys
from ast import literal_eval
from multiprocessing.pool import ThreadPool
lab_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
sys.path.insert(1, lab_root)

import network_builders
from tf_plus import BatchNormalization, Lambda, Dropout
from tf_plus import Conv2D, MaxPooling2D, Flatten, Dense, he_normal, relu, Activation
from tf_plus import Layers, SequentialNetwork, l2reg, PreprocessingLayers
from tf_plus import learning_phase, batchnorm_learning_phase
from exp.tf_nets.losses import add_classification_losses
from brook.tfutil import hist_summaries_train, get_collection_intersection, get_collection_intersection_summary, log_scalars, sess_run_dict
from brook.tfutil import summarize_weights, summarize_opt, tf_assert_all_init, tf_get_uninitialized_variables, add_grad_summaries

def make_parser():
    parser = argparse.ArgumentParser()
    # args that should be the same as train.py:
    parser.add_argument('--train_h5', type=str, required=True)
    parser.add_argument('--test_h5', type=str, required=True)
    parser.add_argument('--input_dim', type=str, default='28,28,1', help='mnist: 28,28,1; cifar: 32,32,3')
    parser.add_argument('--arch', type=str, default='fc', choices=('fc', 'fc_cust', 'lenet', 'allcnn', 'resnet', 'vgg'), help='network architecture')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers for cifar fc')
    parser.add_argument('--opt', type=str, default='sgd', choices=('sgd', 'rmsprop', 'adam'))
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--l2_special', type=float, default=0, help='only used for side resnet experiments')
    parser.add_argument('--resize_more', type=float, default=1, help='only used for side resnet experiments')
    parser.add_argument('--resize_less', type=float, default=1, help='only used for side resnet experiments')

    # adjust batch sizes for cifar:
    parser.add_argument('--large_batch_size', type=int, default=11000, help='mnist: 11000, cifar: 5000')
    parser.add_argument('--test_batch_size', type=int, default=0) # do 0 for all

    # params for calculating gradients:
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=9999999999)
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--default_num_splits', type=int, default=2, help='number of gradient calculations per iteration') # do 2 for RK4
    parser.add_argument('--error_threshold', type=float, default=0.001, help='threshold of error per iter')

    parser.add_argument('--weights_h5', type=str, required=True)
    parser.add_argument('--output_h5', type=str, default=None, help='default = resname/gradients_adaptive')
    parser.add_argument('--stream_inputs', action='store_true', help='read in weights bit by bit')
    return parser

def read_input_data(filename):
    input_file = h5py.File(filename, 'r')
    x = np.array(input_file['images'])
    y = np.array(input_file['labels'])
    input_file.close()
    return x, y

################# model setup, after architecture is already created

def init_model(model, args):
    img_size = tuple([None] + [int(dim) for dim in args.input_dim.split(',')])
    input_images = tf.placeholder(dtype='float32', shape=img_size)
    input_labels = tf.placeholder(dtype='int64', shape=(None,))
    model.a('input_images', input_images)
    model.a('input_labels', input_labels)
    model.a('logits', model(input_images)) # logits is y_pred


def define_training(model, args):
    # define optimizer
    input_lr = tf.placeholder(tf.float32, shape=[]) # placeholder for dynamic learning rate
    model.a('input_lr', input_lr)
    if args.opt == 'sgd':
        optimizer = tf.train.MomentumOptimizer(input_lr, 0) # momentum should not make a difference in calculating gradients
    elif args.opt == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(input_lr)
    elif args.opt == 'adam':
        optimizer = tf.train.AdamOptimizer(input_lr)
    model.a('optimizer', optimizer)

    # This adds prob, cross_ent, loss_cross_ent, class_prediction,
    # prediction_correct, accuracy, loss, (loss_reg) in tf_nets/losses.py
    add_classification_losses(model, model.input_labels)
    model.a('train_step', optimizer.minimize(model.loss))

    # gradients
    grads_and_vars = optimizer.compute_gradients(
        model.loss_cross_ent, model.trainable_weights, model.optimizer.GATE_GRAPH) # change if you want to include l2 loss
    model.a('grads_to_compute', [grad for grad, _ in grads_and_vars])

    # print('All model weights:')
    # summarize_weights(model.trainable_weights)

################# util for training/eval portion

# flatten and concatentate list of tensors into one np vector
def flatten_all(tensors):
    return np.concatenate([tensor.eval().flatten() for tensor in tensors])

# returns list of variables as np arrays in their original shape
def split_and_shape(one_time_slice, shapes):
    variables = []
    offset = 0
    for shape in shapes:
        num_params = np.prod(shape)
        variables.append(one_time_slice[offset : offset + num_params].reshape(shape))
        offset += num_params
    return variables

def get_gradients_and_eval(sess, model, input_x, input_y, dim_sum, batch_size, get_eval=True, get_grads=True):
    grad_sums = np.zeros(dim_sum)
    num_batches = int(input_y.shape[0] / batch_size)
    total_acc = 0
    total_loss = 0
    total_loss_no_reg = 0 # loss without counting l2 penalty

    for i in range(num_batches):
        # slice indices (should be large)
        s_start = batch_size * i
        s_end = s_start + batch_size

        fetch_dict = {}
        if get_eval:
            # fetch_dict['accuracy'] = model.accuracy
            # fetch_dict['loss'] = model.loss
            fetch_dict['loss_no_reg'] = model.loss_cross_ent
        if get_grads:
            fetch_dict['gradients'] = model.grads_to_compute

        result_dict = sess_run_dict(sess, fetch_dict, feed_dict={
            model.input_images: input_x[s_start:s_end],
            model.input_labels: input_y[s_start:s_end],
            learning_phase(): 0,
            batchnorm_learning_phase(): 1})

        if get_eval:
            # total_acc += result_dict['accuracy']
            # total_loss += result_dict['loss']
            total_loss_no_reg += result_dict['loss_no_reg']
        if get_grads:
            grads = result_dict['gradients'] # grads should now be a list of np arrays
            flattened = np.concatenate([grad.flatten() for grad in grads])
            grad_sums += flattened

    acc = total_acc / num_batches
    loss = total_loss / num_batches
    loss_no_reg = total_loss_no_reg / num_batches

    return np.divide(grad_sums, num_batches), loss_no_reg

#################

# loads weights, calculates train and test gradients, writes to file at given iteration
def load_and_calculate(sess, model, weight_values, input_data, dim_sum, args, get_eval=True):
    train_x, train_y, test_x, test_y = input_data
    # load weights into model
    for i, w in enumerate(model.trainable_weights):
        w.load(weight_values[i], session=sess)

    # train set
    cur_train_grads, cur_train_loss = get_gradients_and_eval(sess, model, train_x, train_y,
        dim_sum, args.large_batch_size, get_eval, True)

    # test set
    cur_test_grads, cur_test_loss = get_gradients_and_eval(sess, model, test_x, test_y,
        dim_sum, args.test_batch_size, get_eval, True)

    return cur_train_grads, cur_train_loss, cur_test_grads, cur_test_loss

'''
Loop through all iterations assigned to one gpu.
Every iteration, calculate and save gradients with these steps:
    1. Calculate 2 gradients for rk4 (or use from previous iteration)
    2. Check how inaccurate the loss diff is
    3. While > 0.001, double the number of gradients for this iteration, and check again
    4. Save dynamically sized chunk to file (future work: write in batches if this needs to be sped up)
    5. Also save number of gradients used
'''
def run_thread(gpu, iters_to_calc, all_weights, shapes, input_data, dim_sum, args, dsets, hf_grads):
    # each process writes to a different variable in the file
    grads_train_key = 'grads_train_{}'.format(gpu)
    grads_test_key = 'grads_test_{}'.format(gpu)

    # build model for this process/device
    with tf.device('/device:GPU:{}'.format(gpu)):
        if args.arch == 'fc':
            model = network_builders.build_network_fc(args)
        elif args.arch == 'fc_cust':
            model = network_builders.build_fc_adjustable(args)
        elif args.arch == 'lenet':
            model = network_builders.build_lenet_conv(args)
        elif args.arch == 'allcnn':
            model = network_builders.build_all_cnn(args)
        elif args.arch == 'resnet':
            model = network_builders.build_resnet(args)
        elif args.arch == 'vgg':
            model = network_builders.build_vgg_half(args)
        init_model(model, args)
        define_training(model, args)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())

    grads_train = np.zeros((args.default_num_splits + 1, dim_sum)) # all the ones needed for current iteration rk4
    grads_test = np.zeros((args.default_num_splits + 1, dim_sum))
    newsize = len(iters_to_calc) * args.default_num_splits + 1 # used to resize
    timerstart = time.time()

    # get 0th iteration
    cur_weights_flat = all_weights[iters_to_calc[0]]
    (grads_train[0], cur_train_loss, grads_test[0], cur_test_loss) = load_and_calculate(
        sess, model, split_and_shape(cur_weights_flat, shapes), input_data, dim_sum, args)
    dsets['trainloss'][iters_to_calc[0]] = cur_train_loss
    dsets['testloss'][iters_to_calc[0]] = cur_test_loss
    dsets[grads_train_key][0] = grads_train[0]
    dsets[grads_test_key][0] = grads_test[0]
    grads_ind = 1 # current index to write to for gradients arrays

    for iterations in iters_to_calc:
        # get next iteration gradients and loss, so we have a ground truth loss
        next_weights_flat = all_weights[iterations + 1]
        (grads_train[-1], next_train_loss, grads_test[-1], next_test_loss) = load_and_calculate(
            sess, model, split_and_shape(next_weights_flat, shapes), input_data, dim_sum, args)

        # get the middle fractional iterations
        get_fractional_gradients(range(1, args.default_num_splits), args.default_num_splits, cur_weights_flat,
            next_weights_flat, grads_train, grads_test, input_data, shapes, sess, model, dim_sum, args) #1, or 1,2,3

        # tuple of (train loss diff, test loss diff)
        approx_errors = (calc_approx_error(cur_train_loss, next_train_loss, grads_train, next_weights_flat - cur_weights_flat),
            calc_approx_error(cur_test_loss, next_test_loss, grads_test, next_weights_flat - cur_weights_flat))
        num_splits = args.default_num_splits

        # do smaller splits until error is small enough
        while np.abs(approx_errors).max() > args.error_threshold and num_splits < 32:
            newsize += num_splits # need to resize by this much
            num_splits *= 2
            grads_train_halved = np.zeros((num_splits + 1, dim_sum))
            grads_test_halved = np.zeros((num_splits + 1, dim_sum))
            grads_train_halved[0:num_splits + 1:2] = grads_train # every odd index is zeros
            grads_test_halved[0:num_splits + 1:2] = grads_test

            # get quarter gradients, fill in the rest of grads_train_halved
            get_fractional_gradients(range(1, num_splits, 2), num_splits, cur_weights_flat, next_weights_flat,
                grads_train_halved , grads_test_halved, input_data, shapes, sess, model, dim_sum, args) # 1,3 or 1,3,5,7
            grads_train = grads_train_halved
            grads_test = grads_test_halved
            approx_errors = (calc_approx_error(cur_train_loss, next_train_loss, grads_train, next_weights_flat - cur_weights_flat),
                calc_approx_error(cur_test_loss, next_test_loss, grads_test, next_weights_flat - cur_weights_flat))

        # actually writing to file
        dsets['trainloss'][iterations + 1] = next_train_loss
        dsets['testloss'][iterations + 1] = next_test_loss
        dsets['num_splits'][iterations] = num_splits
        if grads_ind + num_splits > dsets[grads_train_key].shape[0]: # resize when you have to
            dsets[grads_train_key].resize((newsize, dsets[grads_train_key].shape[1]))
            dsets[grads_test_key].resize((newsize, dsets[grads_test_key].shape[1]))
        dsets[grads_train_key][grads_ind:grads_ind + num_splits] = grads_train[1:] # 0 written in previous iteration
        dsets[grads_test_key][grads_ind:grads_ind + num_splits] = grads_test[1:]
        grads_ind += num_splits

        # set variables for next iteration
        cur_weights_flat = next_weights_flat
        cur_train_loss, cur_test_loss = next_train_loss, next_test_loss
        grads_train_new = np.zeros((args.default_num_splits + 1, dim_sum))
        grads_test_new = np.zeros((args.default_num_splits + 1, dim_sum))
        grads_train_new[0], grads_test_new[0] = grads_train[-1], grads_test[-1]
        grads_train, grads_test = grads_train_new, grads_test_new

        if (iterations - iters_to_calc[0]) % args.print_every == 0:
            print('iter {} from gpu {} ({:.2f} s)'.format(iterations, gpu, time.time() - timerstart))

    return gpu

def calc_approx_error(cur_loss, next_loss, gradients, weight_deltas):
    num_terms = gradients.shape[0] # 3 for rk4, 5 for rk8, etc.
    coeffs = np.ones(num_terms)
    coeffs[1::2] = 4
    coeffs[2:-1:2] = 2 # e.g. [1, 4, 2, 4, 1]

    k = np.matmul(coeffs, gradients) / coeffs.sum()
    iol = np.multiply(k, weight_deltas).sum()
    return next_loss - (cur_loss + iol)

# given weights at iterations (i, i+1), number of splits, and which fractions in between,
# calculate and store those gradients
def get_fractional_gradients(fractions, num_splits, cur_weights_flat, next_weights_flat,
                             grads_train, grads_test, input_data, shapes, sess, model, dim_sum, args):
    # fractions should start at 1, not 0
    for frac in fractions:
        next_frac = frac / num_splits
        prev_frac = 1 - next_frac
        # halfway weights, or some other fraction
        partway_weights = split_and_shape(prev_frac * cur_weights_flat + next_frac * next_weights_flat, shapes)

        (grads_train[frac], _, grads_test[frac], _) = load_and_calculate(
            sess, model, partway_weights, input_data, dim_sum, args, get_eval=False)

# divide evenly except last range gets any remainders
def divide_with_remainder(num_iters, num_gpus):
    iters_per_gpu = int(num_iters / num_gpus)
    iters_to_calc = []
    for gpu in range(num_gpus):
        if gpu + 1 == num_gpus:
            iters_to_calc.append(range(gpu * iters_per_gpu, num_iters))
        else:
            iters_to_calc.append(range(gpu * iters_per_gpu, (gpu + 1) * iters_per_gpu))
    return iters_to_calc

def main():
    parser = make_parser()
    args = parser.parse_args()

    # load data
    train_x, train_y = read_input_data(args.train_h5)
    test_x, test_y = read_input_data(args.test_h5) # used as val for now

    images_scale = np.max(train_x)
    if images_scale > 1:
        print('Normalizing images by a factor of {}'.format(images_scale))
        train_x = train_x / images_scale
        test_x = test_x / images_scale
    input_data = (train_x, train_y, test_x, test_y) # package for more concise argument passing

    if args.test_batch_size == 0:
        args.test_batch_size = test_y.shape[0]

    print('Data shapes:', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    if train_y.shape[0] % args.large_batch_size != 0:
        print("WARNING large batch size doesn't divide train set evenly")
    if test_y.shape[0] % args.test_batch_size != 0:
        print("WARNING batch size doesn't divide test set evenly")

    # get all_weights. Do it in 1 chunk if it fits into memory
    hf_weights = h5py.File(args.weights_h5, 'r')
    if args.stream_inputs:
        all_weights = hf_weights['all_weights'] # future work: change to streamds if you want it to be faster
    else:
        all_weights = np.array(hf_weights['all_weights'], dtype='f8')
    shapes = [literal_eval(s) for s in hf_weights.attrs['var_shapes'].decode('utf-8').split(';')]

    print(all_weights.shape)
    print(shapes)
    num_iters = min(args.max_iters, all_weights.shape[0] - 1)
    dim_sum = all_weights.shape[1]

    # set up output file
    output_name = args.output_h5
    if not output_name: # use default gradients name
        assert args.weights_h5[-8:] == '/weights'
        output_name = args.weights_h5[:-8] + '/gradients_adaptive'
        if args.max_iters < all_weights.shape[0] - 1:
            output_name += '_{}iters'.format(args.max_iters)
    print('Writing gradients to file {}'.format(output_name))
    dsets = {}
    hf_grads = h5py.File(output_name, 'w-')
    dsets['trainloss'] = hf_grads.create_dataset('trainloss', (num_iters + 1,), dtype='f4', compression='gzip')
    dsets['testloss'] = hf_grads.create_dataset('testloss', (num_iters + 1,), dtype='f4', compression='gzip')
    dsets['num_splits'] = hf_grads.create_dataset('num_splits', (num_iters,), dtype='i', compression='gzip')

    pool = ThreadPool(args.num_gpus)
    iters_to_calc = divide_with_remainder(num_iters, args.num_gpus)
    results = []
    overall_timerstart = time.time()

    for gpu in range(args.num_gpus):
        # each process writes to a different variable in the file
        dsets['grads_train_{}'.format(gpu)] = hf_grads.create_dataset(
            'grads_train_{}'.format(gpu), (len(iters_to_calc[gpu]) * args.default_num_splits + 1, dim_sum),
            maxshape=(None, dim_sum), dtype='f4', compression='gzip')
        dsets['grads_test_{}'.format(gpu)] = hf_grads.create_dataset(
            'grads_test_{}'.format(gpu), (len(iters_to_calc[gpu]) * args.default_num_splits + 1, dim_sum),
            maxshape=(None, dim_sum), dtype='f4', compression='gzip')

        if args.num_gpus > 1:
            ret = pool.apply_async(run_thread, (gpu, iters_to_calc[gpu], all_weights, shapes, input_data, dim_sum,
                    args, dsets, hf_grads))
            results.append(ret)
        else:
            run_thread(gpu, iters_to_calc[gpu], all_weights, shapes, input_data, dim_sum,
                args, dsets, hf_grads)

    pool.close()
    pool.join()
    print('return values: ', [res.get() for res in results])
    print('total time elapsed:', time.time() - overall_timerstart)
    hf_weights.close()
    hf_grads.close()

if __name__ == '__main__':
    main()
