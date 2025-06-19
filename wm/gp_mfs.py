

'''multiple frequency sinusoidal 用のパラメータを設定'''
# usage:
# import gp_wm as gp
# and use N, bs, etc.

import tensorflow as tf
import numpy as np

task_name = 'mfs'
experiment_name = 'mfs_ld0.02_creg0.0'

freqs = np.array([5.0*np.sqrt(2.0)/1000.0, 5.0 *
          np.sqrt(7.0)/1000.0])  # frequencies
N = 40
bs = 20  # batch_size
alpha = 0.1
scale = 1.0
# use_weight_regularization = False
N_in = 2
N_out = 2
use_global_feedback = False
total_param = N*(N+1) + N_in*N + \
  N_out*(N+1) + N_out*N

internal_noise = 1e-1

rec_initializer = None
# rec_initializer = RandomWithLocalityInitializer(1.0,0.5)

# c_reg = 0.2/N # for l1
#c_reg = 0.5/N  # for l2
######
c_reg = 0.0  # for l2 0.02+locality 0.5 -> success
c_reg_in = 0.0
######
# regularizer = tf.keras.regularizers.l1
# regularizer = None
regularizer = tf.keras.regularizers.l2
learning_rate = 0.002
learning_rate_mine = 0.003

# mine network
N_mt_hidden = 512
g1lim = [0, N//2]
g2lim = [N//2, N]
#####
lambda_mi = 0.02 # weight for mi loss
#####
# time
time_learn = 2000
transient = 1000
split_length = 1000

epochs = 500

initial_mine_epoch = 100
n_mine_pre = 10

clip_grad = 4.0
clip_grad_mine = 4.0
tf.keras.backend.set_floatx('float32')
mydtype = tf.float32
checkpoint_dir = './checkpoints'
