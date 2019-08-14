'''
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
'''

import tensorflow as tf
from .network import SequentialNetwork
from .core import Lambda


class PreprocessingLayers(SequentialNetwork):
    def __init__(self, shift_in=None):
        super(PreprocessingLayers, self).__init__()
        self.add(Lambda(lambda x: tf.to_float(x)))
        if shift_in is not None:
            self.add(Lambda(lambda x: x - shift_in))
