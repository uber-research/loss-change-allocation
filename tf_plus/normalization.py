'''
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
'''

from tensorflow.python.layers.normalization import BatchNormalization as base_BatchNormalization

from .backend import batchnorm_learning_phase


class BatchNormalization(base_BatchNormalization):
    '''Override tf.layers.BatchNormalization with a version that uses
    learning_phase rather than requiring a training={True,False} when
    the layer is called.
    '''

    def __init__(self, *args, **kwargs):
        super(BatchNormalization, self).__init__(*args, **kwargs)
        #self._call_with_training = True
        self._uses_learning_phase = True

    def call(self, inputs, training='do-not-pass-arg', **kwargs):
        assert training == 'do-not-pass-arg', ('This BatchNormalization layer should be called '
                                               'without a training={True,False} arg, because it'
                                               ' sets training using tf_plus.learning_phase()')
        return super(BatchNormalization, self).call(inputs, training=batchnorm_learning_phase(), **kwargs)
