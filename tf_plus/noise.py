'''
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
'''

from tensorflow.python.layers.core import Dropout as base_Dropout

from .backend import learning_phase

class Dropout(base_Dropout):
    '''Override tf.layers.Dropout with a version that uses
    learning_phase rather than requiring a training={True,False} when
    the layer is called.
    '''

    def __init__(self, *args, **kwargs):
        super(Dropout, self).__init__(*args, **kwargs)
        #self._call_with_training = True
        self._uses_learning_phase = True

    def call(self, inputs, training='do-not-pass-arg', **kwargs):
        assert training == 'do-not-pass-arg', ('This Dropout layer should be called '
                                               'without a training={True,False} arg, because it'
                                               'sets training using tf_plus.learning_phase()')
        return super(Dropout, self).call(inputs, training=learning_phase(), **kwargs)
