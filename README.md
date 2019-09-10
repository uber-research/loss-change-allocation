# Loss Change Allocation

## Introduction
This repository contains source code for the experiments in [LCA: Loss Change Allocation for Neural Network Training](https://arxiv.org/abs/1909.01440) (to be presented at NeurIPS 2019) by Janice Lan, Rosanne Liu, Hattie Zhou, and Jason Yosinski.

```
@inproceedings{lan-2019-loss-change-allocation,
  title={LCA: Loss Change Allocation for Neural Network Training},
  author={Janice Lan and Rosanne Liu and Hattie Zhou and Jason Yosinski},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```

## Codebase structure

* `train.py` trains a model and saves weights at every iteration, using architectures defined in `network_builders.py`
* `adaptive_calc_gradients.py` calculates gradients used for LC, based on saved weights
* `save_lc_stream.py` calculates LC for runs where gradients or weights don't fit into memory
* `plot_util.py` demonstrates some examples of various visualizations, aggregations, and analyses

(Optional) This codebase uses the `GitResultsManager` package to keep track of experiments. See: https://github.com/yosinski/GitResultsManager


## Example commands

* To train: `python train.py --train_h5 /data/cifar10_train --test_h5 /data/cifar10_test --input_dim 32,32,3 --arch resnet --opt sgd --lr 0.1 --save_weights --num_epochs 25 --large_batch_size 5000 --test_batch_size 5000 --eval_every 100 --print_every 100 --log_every 50`
* To calculate gradients: `python adaptive_calc_gradients.py --train_h5 /data/cifar10_train --test_h5 /data/cifar10_test --input_dim 32,32,3 --arch resnet --opt sgd --large_batch_size 2500 --test_batch_size 2500 --num_gpus 4 --weights_h5 previous_experiment/weights`
