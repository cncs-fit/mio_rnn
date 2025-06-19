# define global parameter for working memory task

# usage:
# import gp_wm as gp
# and use N, bs, etc.

import tensorflow as tf

task_name = 'wm'
experiment_name = 'wm_ld0.1creg0.01'

N = 40 # number of unit
bs = 20  # batch_size
n_mem = 2 # number of memory
alpha = 0.1 # decay rate
scale = 1.0 # initial scaling of recurrent weight
# use_weight_regularization = False
N_in = n_mem*2
N_out = n_mem
use_global_feedback = False
total_param = N*(N+1) + N_in*N + \
    N_out*(N+1) + N_out*N

internal_noise = 1e-1 # strength of internal noise

rec_initializer = None # initializer for recurrent weight #none means ?
# rec_initializer = RandomWithLocalityInitializer(1.0,0.7)
# c_reg = 1.5/N # for l1
# c_reg = 0.04  # for l2
# c_reg_in = 0.02
#c_reg = 0.03 # loss weight for regularizer
#c_reg_in = 0.03
#################
c_reg = 0.01 # loss weight for regularizer
c_reg_in = 0.01
################

# regularizer = tf.keras.regularizers.l1
# regularizer = None
regularizer = tf.keras.regularizers.l2 # regularizer for recurrent weight
learning_rate = 0.0012 # learning rate
learning_rate = 0.002 # learning rate
learning_rate_mine = 0.003 # learning rate for MINE network

# mine network
N_mt_hidden = 512
g1lim = [0, N//2]
g2lim = [N//2, N]
# lambda_mi = 0.12  # weight for mi loss
###################
###################
lambda_mi = 0.1  # weight for mi loss
####################
####################
# time
time_learn = 2000
transient = 1000
split_length = 2000

epochs = 1000

# for no-MINE
# epochs = 400


initial_mine_epoch = 100
n_mine_pre = 10

clip_grad = 4.0 # clip gradient
clip_grad_mine = 4.0 # clip gradient for mine network

# tf.keras.backend.set_floatx('float32') # set precision
mydtype = tf.float32
checkpoint_dir = './checkpoints'
