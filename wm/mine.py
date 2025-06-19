''' An implementation of MINE: mutual information neural estimation
   for RNN model, using Tensorflow 2.
MINE codes are borrowed from https://github.com/mzgubic/MINE/blob/master/MINE_in_TF.ipynb
and then modified for tf2 implementation
'''
#%%


import numpy as np
import os

from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.util import nest




# class sRNN(keras.Model):
#   ''' simple RNN network just for testing MINE in this file.
#   input -> RNN -> outputp
#   This model keeps RNN's intermediate states at self.X.  
#   '''

#   def __init__(self, N_hidden, N_in,  recurrent_initializer='orthogonal', **kwargs):
#     super(sRNN, self).__init__(**kwargs)
#     self.N_hidden = N_hidden
#     self.N_in = N_in
#     # self.N_out = N_out
#     self.rnn_cell = layers.SimpleRNNCell(
#         N_hidden, recurrent_initializer=recurrent_initializer, bias_initializer='zeros')
#     self.rnn = layers.RNN(self.rnn_cell, return_sequences=True, return_state=True)
#     # self.dense = layers.Dense(N_out, activation=None)
#     # self.sm = layers.Softmax()

#   def call(self, inputs, initial_state=None):
#     # if x0 is None:
#     #   self.X = self.rnn(input_seq) # X.shape=(bs,length,N_hidden)
#     # else:
#     # X.shape=(bs,length,N_hidden)
#     self.X, self.state = self.rnn(inputs, initial_state=initial_state)
#     # self.X_last = self.X[:, -1, :]  # ここで微分とるので，持っておく．
#     return self.X

# T を計算するnet


class MINE_T(layers.Layer):
  ''' T network for MINE (Stat net) used for estimating mutual information.
  '''
  def __init__(self, n_hidden, mydtype=tf.float64, **kwargs):
    '''
    args:
      n_hidden: the number of hidden units
      mydtype: dtype of the network
  
    '''
    super(MINE_T, self).__init__(**kwargs)
    self.dense_x = layers.Dense(n_hidden, dtype=mydtype)
    self.dense_y = layers.Dense(n_hidden, dtype=mydtype)
    self.d2 = layers.Dense(n_hidden, dtype=mydtype)
    self.act1 = layers.ReLU()
    self.act2 = layers.ReLU()
    self.dense_out = layers.Dense(1, dtype=mydtype)

  def call(self, x_in, y_in):
    lx = self.dense_x(x_in) # first dense layer for x
    ly = self.dense_y(y_in) # first dense layer for y
    outputs = self.act1(lx+ly) # ReLU
    outputs = self.d2(outputs) # second dense layer
    outputs = self.act2(outputs) # ReLU

    return self.dense_out(outputs) # last dense layer

# T をラップして MI推定を出す．


class MINE(tf.keras.Model):
  def __init__(self, t_net=None, n_hidden=None, s_noise=1e-5, mydtype=tf.float64, **kwargs):
    super(MINE, self).__init__(**kwargs)
    if t_net is not None:
      self.T = t_net
    else:
      self.T = MINE_T(n_hidden=n_hidden, mydtype=mydtype)

    self.noise_gen = tf.random.get_global_generator()
    self.s_noise = tf.constant(s_noise, dtype=mydtype)
    self.mydtype = mydtype

  def call(self, x_in, y_in):
    ''' 
    args:
      x_in: (sample, dim)
      y_in: (sample, dim)
    '''
    # this tf.random.shuffle is not differentiable.  use tf.gather
    # y_shuffle = tf.random.shuffle(y_in)  # shuffle in the first axis
    y_shuffle = tf.gather(y_in, tf.random.shuffle(tf.range(tf.shape(y_in)[0])))
    x_in = x_in + self.noise_gen.normal(shape=x_in.shape, stddev=self.s_noise, dtype=self.mydtype)
    y_in = y_in + self.noise_gen.normal(shape=x_in.shape, stddev=self.s_noise, dtype=self.mydtype)    
    # y_shuffle = y_shuffle + self.noise_gen.normal(shape=x_in.shape, stddev=self.s_noise, dtype=self.mydtype)        
    # true distribution of XY. output shape is (sample, 1)
    T_xy = self.T(x_in, y_in)
    # pseudo-independent distribution of X and Y made by shuffling
    T_x_y = self.T(x_in, y_shuffle)
    # estimated mi
    return tf.reduce_mean(T_xy, axis=0) - tf.math.log(tf.reduce_mean(tf.math.exp(T_x_y), axis=0))


def separate_group(X, g1lim, g2lim):
  x1 = X[:, :, g1lim[0]:g1lim[1]]
  x2 = X[:, :, g2lim[0]:g2lim[1]]
  return x1, x2



def run_and_extract_group(model, inputs, initial_state, gp):
  ''' run RNN and extract variables of two pre-defined group for MINE
  args:
    model: RNNModel
    inputs: input signal
    initial_state: initial state of RNN
    gp: global parameters
  returns:
    z: output of RNN
    r1: internal states of RNN for group 1. Shape is (bs*t_length, n_g1).
    r2: internal states of RNN for group 2. Shape is (bs*t_length, n_g2).
  '''
  z = model(inputs, initial_state=initial_state)
  r1, r2 = separate_group(model.r, gp.g1lim, gp.g2lim)
  r1 = tf.reshape(r1, [-1, gp.g1lim[1]-gp.g1lim[0]])  # bs, t_length, n_g1 -> bs*t_length, n_g1
  r2 = tf.reshape(r2, [-1, gp.g2lim[1]-gp.g2lim[0]])
  return z, r1, r2


def split_input(input, transient, t_split):
  if transient != 0:
    input_transient = input[:, 0:transient, :]
  else:
    input_transient = None
  input_split = tf.split(input[:, transient:, :],
                         (input.shape[1]-transient)//t_split, axis=1)
  return input_transient, input_split


def _run_and_extract_group(model, input, x0):
  ''' run RNN and extract variables of two pre-defined group for MINE
  '''
  X = model(input, initial_state=x0)
  x1, x2 = separate_group(X, model.g1lim, model.g2lim)
  x1r = tf.reshape(x1, [-1, N_hidden])
  x2r = tf.reshape(x2, [-1, N_hidden])
  return X, x1r, x2r



@tf.function
def run_transient(model, input, x0):
  model(input, initial_state=x0)  # run transient
  return model.state #?




@tf.function
def mine_train_step(model, mine, input, x0, metric_mi=None):
  ''' train statistic network 1 step'''
  X, x1r, x2r = _run_and_extract_group(model, input, x0)
  with tf.GradientTape() as tape:
    mi = mine(x1r, x2r)
    mi_objective = -mi

  gradients = tape.gradient(mi_objective, mine.trainable_variables)
  mine_optimizer.apply_gradients(zip(gradients, mine.trainable_variables))
  if metric_mi is not None:
    metric_mi(mi)
  return X



def mine_train_epoch(model, mine, input, x0, transient, t_split, metric_mi=None):
  ''' train statistic network 1 epoch'''
  input_transient, inputs_split = split_input(input, transient, t_split)

  if transient>0:
    x0 = run_transient(model, input_transient, x0)

  for input_sp in inputs_split:
    X = mine_train_step(model, mine, input_sp, x0, metric_mi)
    x0 = X[:,-1,:]


#%%
if __name__ == '__main__':

  # import matplotlib as mpl
  # if not is_env_notebook():
  #   print('use AGG')
  #   mpl.use('Agg')  # off-screen use of matplotlib
  # import matplotlib.pyplot as plt

  # GPU memory setting
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
      tf.config.experimental.set_memory_growth(physical_devices[k], True)
      print('memory growth:', tf.config.experimental.get_memory_growth(
          physical_devices[k]))
  else:
    print("Not enough GPU hardware devices available")

  tf.keras.backend.set_floatx('float64')
  mydtype = tf.float64
  #%% constants
  figure_dir = 'figures'
  USE_LEAKY = False
  #%%
  if USE_LEAKY:
    N_in = 6
    # N_out = 4
    bs = 50  # batch_size
    transient = 500
    t_length = 4000
    t_split = 500
    total_time = transient + t_length
    N_hidden = 100
    scale = 1.5

    rec_initializer = tf.keras.initializers.TruncatedNormal(
        mean=0.0, stddev=scale/np.sqrt(N_hidden))

    model = LeakyRNNModel(N_hidden, N_in, alpha=0.1, s_noise=0,
                          recurrent_initializer=rec_initializer, dtype=mydtype)
    zero_input = tf.zeros(shape=(bs, t_split, N_in), dtype=mydtype)
    x0 = 0.1*tf.constant(np.random.randn(bs, N_hidden), dtype=mydtype)
    # model is built here.
    model(zero_input, initial_state=x0)  # (bs, t_length, N)
    X = model.X

    model.summary()
    n_param = N_hidden**2 + N_hidden + N_in*N_hidden
    print(f'number of parameters:{n_param}')

  else:
    N_in = 50
    # N_out = 4
    bs = 100  # batch_size
    t_length = 2000
    t_split = 200
    transient = 200
    total_time = transient+t_length
    N_hidden = 100

    scale = 1.5
    rec_initializer = tf.keras.initializers.TruncatedNormal(
        mean=0.0, stddev=scale/np.sqrt(N_hidden))
    # rec_initializer = 'orthogonal'

    model = sRNN(N_hidden, N_in,  recurrent_initializer=rec_initializer)
    model.build(input_shape=[None, None, N_in])
    model.summary()

  #%% run and separate  

  g1_min = 0
  g1_max = N_hidden//2
  g2_min = N_hidden//2
  g2_max = N_hidden
  model.g1lim = (g1_min, g1_max)
  model.g2lim = (g2_min, g2_max)

  rand_g = tf.random.Generator.from_seed(10)

  zero_input = tf.zeros(shape=(bs, total_time, N_in), dtype=mydtype)
  noise_input = rand_g.normal(shape=(bs, total_time, N_in), stddev=0.01, dtype=mydtype)  

  zero_transient, zero_split = split_input(zero_input, transient, t_split)
  noise_transient, noise_split = split_input(noise_input, transient, t_split)

  x0 = 0.1*tf.constant(np.random.randn(bs, N_hidden), dtype=mydtype)
  if transient>0:
    X_trans = model(zero_transient, initial_state=x0)  # (bs, t_length, N)
    x0 = model.state
  X = model(zero_split[0], initial_state=x0)

  x1, x2 = separate_group(X, (g1_min, g1_max), (g2_min, g2_max))
  #%%

  def run_whole_sequence(model, input, x0, transient, t_split):
    Xs=[]
    input_transient, inputs_split = split_input(input, transient, t_split)
    if transient>0:
      X_trans = model(input_transient, initial_state=x0)  # (bs, t_length, N)
      x0 = model.state
    else:
      X_trans = None

    for input_split in inputs_split:
      Xsp = model(input_split, initial_state=x0)
      x0 = model.state
      Xs.append(Xsp)
    X = tf.concat(Xs, axis=1)
    return X, X_trans

  X, X_trans = run_whole_sequence(model, noise_input, x0, transient, t_split)
  if transient>0:
    plt.plot(np.arange(0,transient), X_trans[0,:,0])
  plt.plot(np.arange(transient,transient+500), X[0,0:500,0])  
  #%%

  sum_g1 = np.sum(X.numpy()[:, :, g1_min:g1_max], axis=2)
  sum_g2 = np.sum(X.numpy()[:, :, g2_min:g2_max], axis=2)
  # plt.plot(X[0, :, 0])
  plt.plot(sum_g1[0, 0:200], '.', label='sum g1')
  plt.plot(sum_g2[0, 0:200], '.', label='sum g2')
  plt.xlabel('time step')
  plt.legend()
  plt.savefig(os.path.join(figure_dir, 'before_learning_1'))
  plt.show()
  
  plt.plot(sum_g1[0,:], sum_g2[0,:], '.', markersize=2)

  plt.xlabel(r'$s_{1}$')
  plt.ylabel(r'$s_{2}$')  
  plt.gca().set_aspect('equal')  
  plt.savefig(os.path.join(figure_dir, 'before_learning_s1s2'))

  # %%

  N_mt_hidden = 100  # the number of hidden units

  mine = MINE(n_hidden=N_mt_hidden, s_noise=2.0)

  #%% build mine by run
  x1r = tf.reshape(x1, [-1, N_hidden])
  x2r = tf.reshape(x2, [-1, N_hidden])
  mine(x1r, x2r)
  #%%
  model_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=2.0)
  mine_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=2.0)
  #metrics
  mi_estimate = tf.keras.metrics.Mean(name='mi_estimate')
  mi_estimate_test = tf.keras.metrics.Mean(name='mi_estimate_test')
  mis = []

  # %% make custom training loop


  
  def model_train_epoch(model, mine, input, x0, transient=0, t_split=None, mi_coef=-1.0):
    ''' train main model with MINE (1epoch) '''
    if t_split is None:
      t_split = input.shape[1]
    
    input_transient, inputs_split = split_input(input, transient, t_split)
  
    if transient>0:
      x0 = run_transient(model, input_transient, x0)      

    for input_sp in inputs_split:
      X = model_train_step(model, mine, input_sp, x0, mi_coef=mi_coef)
      x0 = X[:,-1,:]  # 最後の値を次の初期値にする．


  @tf.function
  def model_train_step(model, mine, input, x0, mi_coef=-1.0):
    '''train main model with MINE (1 step) '''
    with tf.GradientTape() as tape:
      X, x1r, x2r = _run_and_extract_group(model, input, x0)      
      mi = mine(x1r, x2r)
      mi_objective = -mi_coef*mi # miを最大化 or 最小化

    gradients = tape.gradient(mi_objective, model.trainable_variables)
    model_optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return X

  @tf.function
  def test_step(model, mine, input, x0, metric_mi):
    X, x1r, x2r = _run_and_extract_group(model, input, x0)          
    mi = mine(x1r, x2r)
    metric_mi(mi)


  def statnet_train(n_epochs, mis=[]):          
    if type(mis) != list:
      mis = mis.tolist()

    mi_estimate.reset_states()    
    for epoch in range(n_epochs):
      ## initial condition
      x0 = 0.1*rand_g.normal(shape=[bs, N_hidden], dtype=mydtype)
      zero_input = tf.zeros(shape=(bs, total_time, N_in))
      noise_input = rand_g.normal(shape=(bs, total_time, N_in), stddev=0.01, dtype=mydtype)  

      mine_train_epoch(model, mine, zero_input, x0, transient=transient, t_split=t_split, metric_mi=mi_estimate)
      # mine_train_step(zero_input, x0,  mi_metric=mi_estimate)

      mis.append(mi_estimate.result())
      print(mis[-1].numpy())
      mi_estimate.reset_states()
    return mis

  def model_train(n_epochs, n_m=20, mi_coef=-1.0, t_split=500,  metric_mi=None, mis=[]):
  
    for epoch in range(n_epochs):
      ## initial condition
      for n in range(n_m):
        mi_estimate.reset_states()
        x0 = 0.1*rand_g.normal(shape=[bs, N_hidden], dtype=mydtype)
        zero_input = tf.zeros(shape=(bs, total_time, N_in))
        mine_train_epoch(model, mine, zero_input, x0, transient=transient, t_split=t_split, metric_mi=metric_mi)

      x0 = 0.1*rand_g.normal(shape=[bs, N_hidden], dtype=mydtype)
      zero_input = tf.zeros(shape=(bs, total_time, N_in))

      model_train_epoch(model, mine, zero_input, x0, mi_coef=mi_coef, t_split=t_split)
      if metric_mi is not None:
        mis.append(metric_mi.result())
      print(mis[-1].numpy())
      mi_estimate.reset_states()
    return mis


  def mi_estimation(model, mine, n_epochs, metric_mi=None):
    metric_mi.reset_states()
    x0 = 0.1*rand_g.normal(shape=[bs, N_hidden], dtype=mydtype)
    zero_input = tf.zeros(shape=(bs, total_time, N_in))
    input_transient, inputs_split = split_input(zero_input, transient, t_split)

    if transient>0:
      x0 = run_transient(model, input_transient, x0)      
    for inputs_s in inputs_split:
      
      test_step(model, mine, inputs_s, x0, metric_mi=metric_mi)
    mi_val = metric_mi.result()
    metric_mi.reset_states()

    return mi_val

  q# %%
  mi_est = []
  #%%  
  mi_est = statnet_train(50, mis=mi_est)
  plt.plot(mi_est)
  #%%  
  mi_est_model = []
  #%%
  mi_est_model = model_train(100, n_m=10, mi_coef=-1.0, metric_mi=mi_estimate, mis=mi_est_model) # mi_coef>0: maximization, mi_coef<0: minimization

  #%%
  plt.plot( range(len(mi_est)),  mi_est)
  plt.plot( range(len(mi_est), len(mi_est)+len(mi_est_model)), mi_est_model     )
  #%%
  X = model(zero_input, initial_state=x0)
  x1, x2 = separate_group(X, model.g1lim, model.g2lim)
  x1r = tf.reshape(x1, [-1, N_hidden])
  x2r = tf.reshape(x2, [-1, N_hidden])


  plt.plot(X[0,0:200, 0], '.')
  plt.show()

  mx1 = tf.reduce_mean(x1,axis=2)
  mx2 = tf.reduce_mean(x2,axis=2)  
  plt.plot(range(200), mx1[0,0:200], mx2[0,0:200])
  plt.show()
  plt.plot( tf.reshape(mx1[:,100:], [-1]),tf.reshape(mx2[:,100:], [-1]), '.')
  #%%
  mival = mi_estimation(model, mine, 100, metric_mi=mi_estimate_test)
  print(f'estimated mi: {mival}')

  # %%
  fig, ax = plt.subplots()
  ax.plot(range(len(mi_est)), mi_est, label='MINE estimate')
  ax.set_xlabel('training steps')
  ax.legend(loc='best')
  #fig.savefig('MINE.png')
  fig.show()
  # %%
