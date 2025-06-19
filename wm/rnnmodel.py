# -*- coding: utf-8 -*-
# define force model
#%%

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, RNN
from tensorflow.keras import Model
import tensorflow.keras.layers as layers
import time
from numpy import linalg as LA  # linear algebra
from tensorflow.python.util import nest





# custom initializer class
class SparseWeightInitializer():
  ''' a variable initializer that makes sparse weight matrix'''

  def __init__(self, p_connect, g_scale, mydtype=tf.float32):
    ''' define connection probability p_connect and scaling constant g_scale
    '''
    self.p_connect = p_connect
    self.g_scale = g_scale
    self.mydtype = mydtype

  def __call__(self, shape):
    c = tf.cast(tf.random.uniform(shape, dtype=self.dtype)
                < self.p_connect, self.mydtype)
    w = c*tf.random.normal(shape, dtype=self.mydtype)

    w = tf.constant(self.g_scale, dtype=self.mydtype) * w / \
        tf.math.sqrt(tf.constant(self.p_connect*shape[0], dtype=self.mydtype))
    return w

class RandomWithLocalityInitializer():
  ''' a variable initializer that makes random weight matrix
      with locality bias
  '''

  def __init__(self, l_sr, sigma_mask, mydtype=tf.float32):
    ''' 
      l_sr: spector radius
    '''
    self.l_sr = l_sr
    self.sigma_mask = sigma_mask
    self.mydtype = mydtype

  def __call__(self, shape, dtype=tf.float32):
    if dtype == None:
      dtype = self.mydtype
    w = tf.random.normal(shape, dtype=dtype) # 乱数
    # Gaussian mask 作成
    N = np.max(shape) 
    ind = tf.reshape( tf.range(0,N, dtype=dtype), (1,N))
    dhat_ij = np.abs(tf.transpose(ind) - ind) # indexの差
    dij = np.minimum(dhat_ij, N-dhat_ij)/N
    print(dij.shape)
    self.mask = tf.math.exp( - dij**2/(self.sigma_mask**2))
    # マスクをかける
    w = w* self.mask 
    # 固有値
    lambdas = tf.linalg.eigvals(w)
    # スペクトル半径
    spr = tf.reduce_max(tf.math.abs(lambdas))
    # スペクトル半径の調整
    w = self.l_sr * w/spr
    return w

#%%

class LeakyRNNCell(layers.SimpleRNNCell):
  '''
  leaky RNN cell inherited from SimpleRNNCell class.
    x_new =  (1-alpha)*x + alpha*( W*tanh(x_prev) +b +W_in*(input)   )
  output = tanh(x_new)
  args:
       units:number of unit
       n_out: dim of output
       alpha=0.1:  dt/tau  tau is time-constant
       s_noise=0.1: noise strength
       use_global_feedback = True : use FORCE model style global feedback element
       gfb_kernel_initializer='glorot_normal',
       output_kernel_initializer='glorot_normal',
       output_bias_initializer='zeros'       
       activation='tanh',
       use_bias=True,
       kernel_initializer='glorot_uniform',
       recurrent_initializer='glorot_normal',
       bias_initializer='zeros',
       kernel_regularizer=None,
       recurrent_regularizer=None,
       bias_regularizer=None,
       kernel_constraint=None,
       recurrent_constraint=None,
       bias_constraint=None,

    
  '''

  def __init__(self,
               units, N_out, alpha=0.1, s_noise=0.0,
               use_global_feedback=True,
               gfb_kernel_initializer='glorot_normal',
               output_kernel_initializer='glorot_normal',
               output_bias_initializer='zeros',
               gfb_kernel_regularizer=None,
               output_kernel_regularizer=None,
               output_bias_regularizer=None,
               activation='tanh',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='glorot_normal',  # random gaussian
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               mydtype=tf.float32,
               **kwargs):

    super(LeakyRNNCell, self).__init__(units, activation=activation,
                                       use_bias=use_bias,
                                       kernel_initializer=kernel_initializer,
                                       recurrent_initializer=recurrent_initializer,
                                       bias_initializer=bias_initializer,
                                       kernel_regularizer=kernel_regularizer,
                                       recurrent_regularizer=recurrent_regularizer,
                                       bias_regularizer=bias_regularizer,
                                       kernel_constraint=kernel_constraint,
                                       recurrent_constraint=recurrent_constraint,
                                       bias_constraint=bias_constraint,
                                       **kwargs)

    self.mydtype = mydtype
    self.N = units
    self.N_out = N_out
    self.alpha = tf.constant(alpha, dtype=mydtype)
    self.om_alpha = tf.constant(1.0-alpha, dtype=mydtype)    # 1.0-alpha
    self.s_noise = tf.constant(s_noise, dtype=mydtype)
    self.noise_g = tf.random.get_global_generator()
    self.use_global_feedback = use_global_feedback


    # get initializers and regularizers for additional weights

    self.output_kernel_initializer = tf.keras.initializers.get(
        output_kernel_initializer)
    self.output_bias_initializer = tf.keras.initializers.get(
        output_bias_initializer)

    self.output_kernel_regularizer = tf.keras.regularizers.get(
        output_kernel_regularizer)
    self.output_bias_regularizer = tf.keras.regularizers.get(
        output_bias_regularizer)

    self.use_global_feedback = use_global_feedback

    self.gfb_kernel_initializer = tf.keras.initializers.get(gfb_kernel_initializer)
    self.gfb_kernel_regularizer = tf.keras.regularizers.get(gfb_kernel_regularizer)

  def build(self, input_shape):
    # print(input_shape)
    # print('model build')    
    super().build(input_shape[0])
    # make additional weights here
    self.output_kernel = self.add_weight(
        shape=(self.N, self.N_out),
        name='output_kernel',
        initializer=self.output_kernel_initializer,
        regularizer=self.output_kernel_regularizer
    )

    self.output_bias = self.add_weight(
        shape=(self.N_out,),
        name='output_bias',
        initializer=self.output_bias_initializer,
        regularizer=self.output_bias_regularizer
    )

    #global_feedback weights
    # if self.use_global_feedback:
    self.gfb_kernel = self.add_weight(
        shape=(self.N_out, self.N),
        name='gfb_kernel',
        initializer=self.gfb_kernel_initializer,
        regularizer=self.gfb_kernel_regularizer
    )
    # print('model build')

  def call(self, inputs, states, training=None):
    prev_x = states[0] if nest.is_nested(states) else states
    input_signal, noise_signal = tf.nest.flatten(inputs)
   
    # input_signal = inputs[0]
    # noise_signal = inputs[1]
    # print(input_signal.shape)
    # print(noise_signal.shape)    

    u = tf.matmul(input_signal, self.kernel) # u = W_in * input
    if self.bias is not None:
      u += self.bias # u+= bias
    r = self.activation(prev_x)
    u += tf.matmul(r, self.recurrent_kernel) # u += W*r ## u = W_in * input + W*r + bias

    # if global feedback is used, output z of previous state is calculated here and then feedback to input
    if self.use_global_feedback:
      z = tf.matmul(r, self.output_kernel) + self.output_bias  # output
      u += tf.matmul(z, self.gfb_kernel)  # adding global feedback term

    x_new = self.om_alpha*prev_x + self.alpha*u + noise_signal # x_new = (1-alpha)*x + alpha*u + noise

    output = x_new  # activation はcellではかけない場合
    # output = self.activation(new_state)  # activation かけたものを出力 する場合
    return output, [x_new]


class LeakyRNNModel(Model):
  def __init__(self, N, N_in=1, N_out=1,
               g_scale=1.2,
               s_noise=0.0,
               alpha=0.1,
               use_global_feedback=True,
               rec_initializer=None,
               c_reg_weight=0.0,
               regularizer=None,
               c_reg_weight_in = 0.0,
               regularizer_in = None,
               mydtype=tf.float32,
               ):
    super(LeakyRNNModel, self).__init__()
    self.N = N
    self.N_in = N_in
    self.N_out = N_out
    self.g_scale = g_scale
    self.s_noise = s_noise
    self.alpha = alpha
    self.use_global_feedback = use_global_feedback
    self.c_reg_weight = c_reg_weight
    self.c_reg_weight_in = c_reg_weight_in
    self.noise_g = tf.random.Generator.from_non_deterministic_state()    
    self.mydtype = mydtype

    #setting initializer
    self.g_scale = g_scale
    if rec_initializer == None:
      rec_initializer = tf.keras.initializers.RandomNormal(
          mean=0.0, stddev=g_scale/np.sqrt(N))

    # setting regularizer
    if regularizer != None:
      self.use_regularizer = True
      reg_rec = regularizer(self.c_reg_weight) # regularizer for recurrent weight
      reg_in = regularizer(self.c_reg_weight_in) # regularizer for input weight
      reg_out = regularizer(self.c_reg_weight) # regularizer for output weight
      if self.use_global_feedback:
        reg_gfb = regularizer(self.c_reg_weight_in)
      else:
        reg_gfb = regularizer(0.0)
    else:
      self.use_regularizer = False      
      reg_rec = None
      reg_in = None
      reg_out = None
      reg_gfb = None


    self.rnn_cell = LeakyRNNCell(units=N, N_out=N_out, alpha=alpha, s_noise=s_noise,
                                 use_global_feedback=self.use_global_feedback,
                                 kernel_initializer='glorot_uniform',
                                 gfb_kernel_initializer='glorot_uniform',
                                 output_kernel_initializer='glorot_uniform',
                                 output_bias_initializer='zeros',
                                 recurrent_initializer=rec_initializer,
                                 gfb_kernel_regularizer=reg_gfb,
                                 recurrent_regularizer=reg_rec,
                                 kernel_regularizer=reg_in,
                                 output_kernel_regularizer=reg_out,
                                 mydtype=mydtype,
                                 )

    self.rnn = layers.RNN(
        self.rnn_cell, return_sequences=True, return_state=True)

  def call(self, inputs, initial_state=None, training=False):
    ''' defines model input-output
    '''
    # noise generation
    self.noise_input = self.s_noise * self.noise_g.normal(shape=[inputs.shape[0], inputs.shape[1], self.N])

    # run rnn. output is (x(internal state), state)
    self.x, self.last_state = self.rnn((inputs, self.noise_input), initial_state=initial_state)
    # r is output of units
    self.r = self.rnn_cell.activation(self.x)
    # output z was re-calculated from r. its already calculated in rnn_cell.call but not taken out.
    self.z = tf.matmul(self.r, self.rnn_cell.output_kernel) + \
        self.rnn_cell.output_bias
    return self.z

  def additional_setup(self):
    ''' set set convenient name for important weights'''
    self.W_rec = self.rnn_cell.recurrent_kernel
    self.W_in = self.rnn_cell.kernel
    self.bias = self.rnn_cell.bias
    self.W_out = self.rnn_cell.output_kernel
    self.b_out = self.rnn_cell.output_bias
    self.W_fb = self.rnn_cell.gfb_kernel



def build_rnn_model(gp):
  '''
  building rnn model
  args: gp: global parameters
  returns model
  modelを作るときに利用．複数のファイルで生成方法を統一するため．'''
  return LeakyRNNModel(N=gp.N, N_in=gp.N_in, N_out=gp.N_out, g_scale=gp.scale, alpha=gp.alpha,
                       s_noise=gp.internal_noise, use_global_feedback=gp.use_global_feedback,
                       rec_initializer=gp.rec_initializer,
                       c_reg_weight=gp.c_reg, c_reg_weight_in=gp.c_reg_in, regularizer=gp.regularizer, mydtype=gp.mydtype)


@tf.function
def transient(model, inputs_transient, initial_state=None):
  ''' 
    初期状態を与えて，transientを走らせる. last_stateなどを返す
    args:
      model: RNNModel
      inputs_transient: inputs for transient period
      initial_state: initial state of RNN
    returns:
      z_trans: outputs in transient period
      r: internal state in transient period
      last_state: last state of RNN

  '''
  if inputs_transient.shape[1] > 0:
    #i_trans, te_trans = task.signals_transient()
    z_trans = model(inputs_transient, initial_state=initial_state)
    return z_trans, model.r, model.last_state
  else:
    return None, None

#%%
if __name__ == '__main__':
  bs = 10
  N = 200
  N_in =2
  N_out = 3
  tmax = 5000
  tf.keras.backend.set_floatx('float32')
  mydtype=tf.float32

  model = LeakyRNNModel(N=N, N_in=N_in, N_out=N_out, g_scale=1.4, alpha=0.1, s_noise=1e-2,
                        c_reg_weight=0.1, regularizer=tf.keras.regularizers.l1, mydtype=mydtype)
  input = tf.zeros(shape=[bs, tmax , N_in], dtype=mydtype)
  x0 = tf.random.uniform(shape=[bs,N], dtype=mydtype)

  #%%
  model.rnn_cell.build([[bs,N_in], [bs, N]] )
  #%%
  z = model(input, initial_state=x0)
  model.additional_setup()

  n_param = N*N + N +N_in*N + N*N_out + N_out + N_out*N
  print(n_param)
  
  print(z.shape)
  print(model.r.shape)
  print(model.z.shape)
  model.summary()
  W = model.W_rec.numpy()
  L = np.linalg.eigvals(W)
  print(np.abs(L)[0])
  plt.plot(z[0,:,0:30])


  # %%


  # %%
  rwi = RandomWithLocalityInitializer(1.0, 1.0) 
  w = rwi([40,40])
  # %%
  plt.imshow(np.abs(w))
  # %%


  lambdas = tf.linalg.eigvals(w)
    # スペクトル半径
  spr = tf.reduce_max(tf.math.abs(lambdas))

  # %%
