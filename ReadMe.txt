Main Changes (arguments):
LIKELIHOOD_DS_BNN: 5.0 => 1.0
LR_BNN: 1e-4 => 1e-2
sample_pool_size: 10 => 5000
MAX_EPOCH_BNN: 1000 => 5

Main Changes (functionality):
Enable training policy net on GPU
Normalize test set if train set is normalized using same statistics
Record detailed and informative values using tensorboard
N of hidden units of policy net can be specified now

Arguments to be tuned for different applications:
ETA
LR_BNN
EXTRA_W_KL

(Other arguments may also need to be tuned, but the effect of the listed ones could be more explainable and significant)
