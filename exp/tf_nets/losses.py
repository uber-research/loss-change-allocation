'''
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
'''

import tensorflow as tf
from brook.tfutil import hist_summaries_traintest, scalar_summaries_traintest


def add_classification_losses(model, input_labels, l2=0):
    '''Add classification and L2 losses'''

    with tf.name_scope('losses'):
        model.a('prob', tf.nn.softmax(model.logits, name='prob'))
        model.a('cross_ent', tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model.logits, labels=input_labels, name='cross_ent'))
        model.a('loss_cross_ent', tf.reduce_mean(model.cross_ent, name='loss_cross_ent'), trackable=True)
        model.a('class_prediction', tf.argmax(model.prob, 1))

        model.a('prediction_correct', tf.equal(model.class_prediction, input_labels, name='prediction_correct'))
        model.a('accuracy', tf.reduce_mean(tf.to_float(model.prediction_correct), name='accuracy'), trackable=True)
        hist_summaries_traintest(model.prob, model.cross_ent)
        #scalar_summaries_traintest(loss_cross_ent, loss_spring, loss, accuracy)
        scalar_summaries_traintest(model.accuracy)

        reg_losses = model.losses
        if len(reg_losses) > 0:
            model.a('loss_reg', tf.add_n(reg_losses, name='reg_loss'), trackable=True)
            model.a('loss', tf.add(model.loss_cross_ent, model.loss_reg, name='loss'), trackable=True)
        else:
            model.a('loss', model.loss_cross_ent, trackable=True)
