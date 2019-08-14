
from __future__ import print_function
from __future__ import division

from ast import literal_eval
import tensorflow as tf
import numpy as np
import time
import h5py
import argparse
import os
import sys
lab_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
sys.path.insert(1, lab_root)

import network_builders
import masked_networks
from tf_plus import BatchNormalization, Lambda, Dropout
from tf_plus import Conv2D, MaxPooling2D, Flatten, Dense, he_normal, relu, Activation
from tf_plus import Layers, SequentialNetwork, l2reg, PreprocessingLayers
from tf_plus import learning_phase, batchnorm_learning_phase
from exp.tf_nets.losses import add_classification_losses
from brook.tfutil import hist_summaries_train, get_collection_intersection, get_collection_intersection_summary, log_scalars, sess_run_dict
from brook.tfutil import summarize_weights, summarize_opt, tf_assert_all_init, tf_get_uninitialized_variables, add_grad_summaries

from IPython import embed

def make_parser():
    parser = argparse.ArgumentParser()
    # inputs
    parser.add_argument('--train_h5', type=str, required=True)
    parser.add_argument('--test_h5', type=str, required=True)
    parser.add_argument('--input_dim', type=str, default='28,28,1', help='mnist: 28,28,1; cifar: 32,32,3')
    parser.add_argument('--init_weights_h5', type=str, required=False)

    # model architecture
    parser.add_argument('--arch', type=str, default='fc', choices=('fc', 'fc_mask', 'conv2_mask', 'conv4_mask',
        'conv6_mask', 'fc_cust', 'lenet', 'allcnn', 'resnet', 'vgg'), help='network architecture')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers for cifar fc')

    # training params
    parser.add_argument('--opt', type=str, default='sgd', choices=('sgd', 'rmsprop', 'adam'))
    parser.add_argument('--lr', type=float, default=.01, help='suggested: .01 sgd, .001 rmsprop, .0001 adam')
    parser.add_argument('--decay_schedule', type=str, default='-1', help='comma separated decay learning rate. allcnn: 200,250,300')
    parser.add_argument('--mom', type=float, default=.9, help='momentum (only has effect for sgd/rmsprop)')
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--train_batch_size', type=int, default=250)
    parser.add_argument('--large_batch_size', type=int, default=11000, help='use mnist: 11000, cifar: 5000')
    parser.add_argument('--test_batch_size', type=int, default=0) # do 0 for all
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--shuffle_seed', type=int, default=-1, help='seed if you want to shuffle batches')
    parser.add_argument('--tf_seed', type=int, default=-1, help='tensorflow random seed')

    # eval and outputs
    parser.add_argument('--print_every', type=int, default=100, help='print status update every n iterations')
    parser.add_argument('--output_dir', type=str, default=os.environ.get('GIT_RESULTS_MANAGER_DIR', None), help='output directory')
    parser.add_argument('--eval_every', type=int, default=20, help='eval on entire set')
    parser.add_argument('--log_every', type=int, default=5, help='save tb batch acc/loss every n iterations')
    parser.add_argument('--save_weights', action='store_true', help='save weights to file')
    parser.add_argument('--save_training_grads', action='store_true', help='save mini-batch gradients to file')
    parser.add_argument('--save_every', type=int, default=1, help='save gradients every n iterations (averaged)') # kinda deprecated

    # special configs for side experiments
    parser.add_argument('--l2_special', type=float, default=0, help='only used for side resnet experiments')
    parser.add_argument('--resize_more', type=float, default=1, help='only used for side resnet experiments')
    parser.add_argument('--resize_less', type=float, default=1, help='only used for side resnet experiments')
    parser.add_argument('--freeze_layers', type=str, default='', help='comma separated layer indices to freeze')
    parser.add_argument('--freeze_starting', type=int, default=0, help='what iteration to start freezing specified layers')
    parser.add_argument('--opt2_layers', type=str, default='', help='comma separated layer indices to use different optimizer on')
    parser.add_argument('--lr_special', type=float, default=.1, help='learning rate for specified layers')
    parser.add_argument('--mom_special', type=float, default=.9, help='momentum for specified layers')

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
        optimizer = tf.train.MomentumOptimizer(input_lr, args.mom)
    elif args.opt == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(input_lr, momentum=args.mom)
    elif args.opt == 'adam':
        optimizer = tf.train.AdamOptimizer(input_lr)
    model.a('optimizer', optimizer)

    # This adds prob, cross_ent, loss_cross_ent, class_prediction,
    # prediction_correct, accuracy, loss, (loss_reg) in tf_nets/losses.py
    add_classification_losses(model, model.input_labels)

    model.a('train_step', optimizer.minimize(model.loss, var_list=model.trainable_weights))
    if len(args.freeze_layers) > 0: # freeze certain layers
        inds_to_ignore = [int(i) for i in args.freeze_layers.split(',')]
        inds_to_train = np.delete(np.arange(len(model.trainable_weights)), inds_to_ignore)
        print('Only training layers:', inds_to_train)
        model.a('train_step_freeze', optimizer.minimize(model.loss, var_list=[model.trainable_weights[i] for i in inds_to_train]))

    if len(args.opt2_layers) > 0: # use 2 different optimizers for different layers
        # This only uses a constant LR, no decaying LR
        if args.opt == 'sgd':
            optimizer_special = tf.train.MomentumOptimizer(args.lr_special, args.mom_special)
        elif args.opt == 'rmsprop':
            optimizer_special = tf.train.RMSPropOptimizer(args.lr_special, momentum=args.mom_special)
        elif args.opt == 'adam':
            optimizer_special = tf.train.AdamOptimizer(args.lr_special)
        model.a('optimizer_special', optimizer_special)

        inds_for_opt2 = [int(i) for i in args.opt2_layers.split(',')]
        inds_for_opt1 = np.delete(np.arange(len(model.trainable_weights)), inds_for_opt2)
        print('These layers get opt1:', inds_for_opt1)
        print('These layers get opt2:', inds_for_opt2)
        model.a('train_step_1', optimizer.minimize(model.loss, var_list=[model.trainable_weights[i] for i in inds_for_opt1]))
        model.a('train_step_2', optimizer_special.minimize(model.loss, var_list=[model.trainable_weights[i] for i in inds_for_opt2]))

    # gradients (may be used for mini-batches)
    grads_and_vars = optimizer.compute_gradients(model.loss, model.trainable_weights)
    model.a('grads_to_compute', [grad for grad, _ in grads_and_vars])

    print('All model weights:')
    summarize_weights(model.trainable_weights)
    print('trainable:')
    for x in model.trainable_weights:
        print(x)
    print('non trainable:')
    for x in model.non_trainable_weights:
        print(x)
    print('saved to weights file:')
    for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print(x)
    # print('grad summaries:')
    # add_grad_summaries(grads_and_vars)
    # print('opt summary:')
    # summarize_opt(optimizer)

################# methods used for freezing layers

# returns list of variables as np arrays in their original shape
def split_and_shape(one_time_slice, shapes):
    variables = []
    offset = 0
    for shape in shapes:
        num_params = np.prod(shape)
        variables.append(one_time_slice[offset : offset + num_params].reshape(shape))
        offset += num_params
    return variables

def load_initial_weights(sess, model, args):
    hf_weights = h5py.File(args.init_weights_h5, 'r')
    init_weights_flat = hf_weights.get('all_weights')[0]
    shapes = [literal_eval(s) for s in hf_weights.attrs['var_shapes'].decode('utf-8').split(';')]
    hf_weights.close()

    weight_values = split_and_shape(init_weights_flat, shapes)
    for i, w in enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)):
        print('loading weights for layer {}: {}'.format(i, w.name))
        w.load(weight_values[i], session=sess)

################# util for training/eval portion

# flatten and concatentate list of tensors into one np vector
def flatten_all(tensors):
    return np.concatenate([tensor.eval().flatten() for tensor in tensors])

# eval on whole train/test set occasionally, for tuning purposes
def eval_on_entire_dataset(sess, model, input_x, input_y, dim_sum, batch_size, tb_prefix, tb_writer, iterations):
    grad_sums = np.zeros(dim_sum)
    num_batches = int(input_y.shape[0] / batch_size)
    total_acc = 0
    total_loss = 0
    total_loss_no_reg = 0 # loss without counting l2 penalty

    for i in range(num_batches):
        # slice indices (should be large)
        s_start = batch_size * i
        s_end = s_start + batch_size

        fetch_dict = {
                'accuracy': model.accuracy,
                'loss': model.loss,
                'loss_no_reg': model.loss_cross_ent}

        result_dict = sess_run_dict(sess, fetch_dict, feed_dict={
            model.input_images: input_x[s_start:s_end],
            model.input_labels: input_y[s_start:s_end],
            learning_phase(): 0,
            batchnorm_learning_phase(): 1}) # do not use nor update moving averages

        total_acc += result_dict['accuracy']
        total_loss += result_dict['loss']
        total_loss_no_reg += result_dict['loss_no_reg']

    acc = total_acc / num_batches
    loss = total_loss / num_batches
    loss_no_reg = total_loss_no_reg / num_batches

    # tensorboard
    if tb_writer:
        summary = tf.Summary()
        summary.value.add(tag='%s_acc' % tb_prefix, simple_value=acc)
        summary.value.add(tag='%s_loss' % tb_prefix, simple_value=loss)
        summary.value.add(tag='%s_loss_no_reg' % tb_prefix, simple_value=loss_no_reg)
        tb_writer.add_summary(summary, iterations)

    return acc, loss_no_reg

#################

def train_and_eval(sess, model, train_x, train_y, test_x, test_y, tb_writer, dsets, args):
    # constants
    num_batches = int(train_y.shape[0] / args.train_batch_size)
    print('Training batch size {}, number of iterations: {} per epoch, {} total'.format(
        args.train_batch_size, num_batches, args.num_epochs*num_batches))
    dim_sum = sum([tf.size(var).eval() for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])

    # adaptive learning schedule
    curr_lr = args.lr
    decay_epochs = [int(ep) for ep in args.decay_schedule.split(',')]
    if decay_epochs[-1] > 0:
        decay_epochs.append(-1) # end with something small to stop the decay
    decay_count = 0

    # initializations
    tb_summaries = tf.summary.merge(tf.get_collection('tb_train_step'))
    shuffled_indices = np.arange(train_y.shape[0]) # for no shuffling
    iterations = 0
    chunks_written = 0 # for args.save_every batches
    timerstart = time.time()

    for epoch in range(args.num_epochs):
        # print('-' * 100)
        # print('epoch {}  current lr {:.3g}'.format(epoch, curr_lr))
        if not args.no_shuffle:
            shuffled_indices = np.random.permutation(train_y.shape[0]) # for shuffled mini-batches

        if epoch == decay_epochs[decay_count]:
            curr_lr *= 0.1
            decay_count += 1

        for i in range(num_batches):
            # store current weights and gradients
            if args.save_weights and iterations % args.save_every == 0:
                dsets['all_weights'][chunks_written] = flatten_all(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
                chunks_written += 1

            # less frequent, larger evals
            if iterations % args.eval_every == 0:
                # eval on entire train set
                cur_train_acc, cur_train_loss = eval_on_entire_dataset(sess, model, train_x, train_y,
                        dim_sum, args.large_batch_size, 'eval_train', tb_writer, iterations)

                # eval on entire test/val set
                cur_test_acc, cur_test_loss = eval_on_entire_dataset(sess, model, test_x, test_y,
                        dim_sum, args.test_batch_size, 'eval_test', tb_writer, iterations)

            # print status update
            if iterations % args.print_every == 0:
                print(('{}: train acc = {:.4f}, test acc = {:.4f}, '
                    + 'train loss = {:.4f}, test loss = {:.4f} ({:.2f} s)').format(iterations,
                    cur_train_acc, cur_test_acc, cur_train_loss, cur_test_loss, time.time() - timerstart))

            # current slice for input data
            batch_indices = shuffled_indices[args.train_batch_size * i : args.train_batch_size * (i + 1)]

            # training
            # fetch_dict = {'accuracy': model.accuracy, 'loss': model.loss} # no longer used
            if len(args.freeze_layers) > 0 and iterations >= args.freeze_starting:
                fetch_dict = {'train_step': model.train_step_freeze}
            elif len(args.opt2_layers) > 0:
                fetch_dict = {'train_step_1': model.train_step_1,
                    'train_step_2': model.train_step_2}
            else:
                fetch_dict = {'train_step': model.train_step}
            fetch_dict.update(model.update_dict())

            if iterations % args.log_every == 0:
                fetch_dict.update({'tb': tb_summaries})
            if args.save_training_grads:
                fetch_dict['gradients'] = model.grads_to_compute

            result_train = sess_run_dict(sess, fetch_dict, feed_dict={
                model.input_images: train_x[batch_indices],
                model.input_labels: train_y[batch_indices],
                model.input_lr: curr_lr,
                learning_phase(): 1,
                batchnorm_learning_phase(): 1})

            # log to tensorboard
            if tb_writer and iterations % args.log_every == 0:
                tb_writer.add_summary(result_train['tb'], iterations)

            if args.save_training_grads:
                dsets['training_grads'][iterations] = np.concatenate(
                    [grad.flatten() for grad in result_train['gradients']])

            iterations += 1

    # save final weight values
    if args.save_weights:
        dsets['all_weights'][chunks_written] = flatten_all(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    # save final evals
    if iterations % args.eval_every == 0:
        # on entire train set
        cur_train_acc, cur_train_loss = eval_on_entire_dataset(sess, model, train_x, train_y,
            dim_sum, args.large_batch_size, 'eval_train', tb_writer, iterations)

        # on entire test/val set
        cur_test_acc, cur_test_loss = eval_on_entire_dataset(sess, model, test_x, test_y,
            dim_sum, args.test_batch_size, 'eval_test', tb_writer, iterations)

    # print last status update
    print(('{}: train acc = {:.4f}, test acc = {:.4f}, '
        + 'train loss = {:.4f}, test loss = {:.4f} ({:.2f} s)').format(iterations,
        cur_train_acc, cur_test_acc, cur_train_loss, cur_test_loss, time.time() - timerstart))


def main():
    parser = make_parser()
    args = parser.parse_args()

    if args.tf_seed != -1:
        tf.random.set_random_seed(args.tf_seed)
    if not args.no_shuffle and args.shuffle_seed != -1:
        np.random.seed(args.shuffle_seed)

    # load data
    train_x, train_y = read_input_data(args.train_h5)
    test_x, test_y = read_input_data(args.test_h5) # used as val

    images_scale = np.max(train_x)
    if images_scale > 1:
        print('Normalizing images by a factor of {}'.format(images_scale))
        train_x = train_x / images_scale
        test_x = test_x / images_scale

    if args.test_batch_size == 0:
        args.test_batch_size = test_y.shape[0]

    print('Data shapes:', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    if train_y.shape[0] % args.train_batch_size != 0:
        print("WARNING batch size doesn't divide train set evenly")
    if train_y.shape[0] % args.large_batch_size != 0:
        print("WARNING large batch size doesn't divide train set evenly")
    if test_y.shape[0] % args.test_batch_size != 0:
        print("WARNING batch size doesn't divide test set evenly")

    # build model
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
    else:
        raise Error("Unknown architeciture {}".format(args.arch))

    init_model(model, args)
    define_training(model, args)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    if args.init_weights_h5:
        load_initial_weights(sess, model, args)

    for collection in ['tb_train_step']: # 'eval_train' and 'eval_test' added manually later
        tf.summary.scalar(collection + '_acc', model.accuracy, collections=[collection])
        tf.summary.scalar(collection + '_loss', model.loss, collections=[collection])

    tb_writer, hf = None, None
    dsets = {}
    if args.output_dir:
        tb_writer = tf.summary.FileWriter(args.output_dir, sess.graph)
        # set up output for gradients/weights
        if args.save_weights:
            dim_sum = sum([tf.size(var).eval() for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
            total_iters = args.num_epochs * int(train_y.shape[0] / args.train_batch_size)
            total_chunks = int(total_iters / args.save_every)
            hf = h5py.File(args.output_dir + '/weights', 'w-')

            # write metadata
            var_shapes = np.string_(';'.join([str(var.get_shape()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]))
            hf.attrs['var_shapes'] = var_shapes
            var_names = np.string_(';'.join([str(var.name) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]))
            hf.attrs['var_names'] = var_names

            # all individual weights at every iteration, where all_weights[i] = weights before iteration i:
            dsets['all_weights'] = hf.create_dataset('all_weights', (total_chunks + 1, dim_sum), dtype='f8', compression='gzip')
        if args.save_training_grads:
            dsets['training_grads'] = hf.create_dataset('training_grads', (total_chunks, dim_sum), dtype='f8', compression='gzip')

    ########## Run main thing ##########
    print('=' * 100)
    train_and_eval(sess, model, train_x, train_y, test_x, test_y, tb_writer, dsets, args)

    if tb_writer:
        tb_writer.close()
    if hf:
        hf.close()

if __name__ == '__main__':
    main()

