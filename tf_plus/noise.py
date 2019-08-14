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
