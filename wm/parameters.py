#%%
import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
from rnnmodel import LeakyRNNModel, RandomWithLocalityInitializer
# TASK_NAME = 'wm'
TASK_NAME = 'mfs'
TASK_NAME = 'rossler'
TASK_NAME = 'wm'
tf.config.experimental.enable_tensor_float_32_execution(
    False)  # tf32 (混合精度)を使わない

#tf.keras.backend.set_floatx('float32')
# mydtype = tf.float32


def set_randseed(rand_seed=1219):
  tf.random.set_seed(seed=rand_seed)
  np.random.seed(seed=rand_seed)


def set_gp_rossler():
  ''' Rossler task用のパラメータを設定'''
  gp = GlobalParameter()
  gp.task_name = 'rossler'
  gp.N = 100
  gp.bs = 50  # batch_size
  gp.alpha = 0.1
  gp.scale = 1.0  # initial weight scaling
  # gp.use_weight_regularization = False
  gp.N_in = 3
  gp.N_out = 3
  gp.use_global_feedback = False

  gp.total_param = gp.N*(gp.N+1) + gp.N_in*gp.N + \
      gp.N_out*(gp.N+1) + gp.N_out*gp.N

  gp.internal_noise = 1e-2

  gp.rec_initializer = None
  # gp.rec_initializer = RandomWithLocalityInitializer(1.0,0.5)

  # gp.c_reg = 0.2/gp.N # for l1
  #gp.c_reg = 0.5/gp.N  # for l2
  gp.c_reg = 0.02  # for l2 0.02+locality 0.5 -> success
  gp.c_reg_in = 0.02
  # gp.regularizer = tf.keras.regularizers.l1
  # gp.regularizer = None
  gp.regularizer = tf.keras.regularizers.l2
  gp.learning_rate = 0.0012
  gp.learning_rate_mine = 0.003

  # mine network
  gp.N_mt_hidden = 512
  gp.g1lim = [0, gp.N//2]
  gp.g2lim = [gp.N//2, gp.N]

  gp.lambda_mi = 0.04  # weight for mi loss

  # time
  gp.time_learn = 2000
  gp.transient = 1000
  gp.split_length = 1000

  gp.epochs = 400

  gp.initial_mine_epoch = 200
  gp.n_mine_pre = 20

  gp.clip_grad = 4.0
  gp.clip_grad_mine = 4.0
  tf.keras.backend.set_floatx('float32')
  gp.mydtype = tf.float32

  # rossler parameters
  gp.ros_dt = 0.1  # time-step
  gp.ros_pred_step = 1
  gp.ros_params = np.array([0.2, 2.0, 5.7])  # a, b, and c

  return gp


class MSEPerVariable(tf.keras.metrics.Metric):
  ''' ターゲット変数ごとにMSEをとるカスタムのメトリッククラス
  '''
  def __init__(self, n_var, name='mse_per_variable'):
    super().__init__(name)
    self.sum_se_pv = self.add_weight(name='sum_se_pv', initializer='zeros', shape=[n_var]) # sum of squared error
    self.n_samples = self.add_weight(name='n_samples', initializer='zeros') # number of samples
    # self.mse_func = tf.keras.losses.MeanSquaredError()

  def update_state(self, y_true, y_pred):
    ''' assuming y_true and y_pred are 3D array (batch, time, dim_variable)
    '''
    y_true = tf.cast(y_true, tf.float32)
    sum_se = tf.reduce_sum(tf.square(y_true - y_pred), axis=[0,1]) #(dim_variable,)
    self.sum_se_pv.assign_add(sum_se)
    self.n_samples.assign_add(tf.cast(tf.shape(y_true)[0]*tf.shape(y_true)[1], tf.float32))


  def result(self):
    return self.sum_se_pv / self.n_samples

  def reset_states(self):
    self.sum_se_pv.assign(tf.zeros_like(self.sum_se_pv))
    self.n_samples.assign(tf.zeros_like(self.n_samples))
                          


class BinaryAccuracyPerVariable(tf.keras.metrics.Metric):
  ''' ターゲット変数ごとにAccuracyをとるカスタムのメトリッククラス
  '''
  def __init__(self, n_var, name='acc_per_variable'):
    super().__init__(name)
    self.n_correct_pv = self.add_weight(name='n_correct_pv', initializer='zeros', shape=[n_var]) # number of correct samples
    self.n_total = self.add_weight(name='n_total', initializer='zeros')   # total number of samples


  def update_state(self, y_true, y_pred, threshold=0.5):
    ''' assuming y_true and y_pred are 3D array (batch, time, dim_variable)
    args:
      y_true: 3D array (batch, time, dim_variable)
      y_pred: 3D array (batch, time, dim_variable)
      threshold: float

    '''
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    acc = tf.reduce_sum(tf.cast(y_true == y_pred, tf.float32), axis=[0,1]) #(dim_variable)
    self.n_correct_pv.assign_add(acc)
    self.n_total.assign_add(tf.cast(tf.shape(y_true)[0]*tf.shape(y_true)[1], tf.float32))    


  def result(self):
    # return ratio of correct samples
    return self.n_correct_pv / self.n_total

  def reset_states(self):
    ''' reset n_correct and n_total
    '''
    self.n_correct_pv.assign(tf.zeros_like(self.n_correct_pv))
    self.n_total.assign(tf.zeros_like(self.n_total))


class WMMetrics:
  def __init__(self, n_var=2):
    self.train_loss = tf.keras.metrics.Mean(name='train_loss') # task loss
    self.train_loss_each = MSEPerVariable(n_var, name='train_loss_each') # task loss    
    self.reg_loss = tf.keras.metrics.Mean(name='regularizer_loss')
    self.total_loss = tf.keras.metrics.Mean(name='total_loss')
    self.train_acc = tf.keras.metrics.BinaryAccuracy(name='accuracy')
    self.train_acc_each = BinaryAccuracyPerVariable(n_var, name='accuracy_each') #  accuracy for each variable
    self.total_grad = tf.keras.metrics.Mean(name='total_grad')

    self.mine_loss = tf.keras.metrics.Mean(name='train_mine_loss')

  def get_metric(self, model:Model, loss, weight_loss, total_loss, teacher, z_tr, norm_grad_all)->None:
    self.train_loss(loss)
    self.train_loss_each(teacher, z_tr)
    if model.use_regularizer:
      self.reg_loss(weight_loss)
    self.total_loss(total_loss)
    self.train_acc(tf.round((teacher+1.0)/2.0),
                   (z_tr+1.0)/2.0)  # 
    self.train_acc_each(tf.round((teacher+1.0)/2.0),
                   (z_tr+1.0)/2.0)  # 

    self.total_grad(norm_grad_all)

  def reset_all_states(self):
    self.train_loss.reset_states()
    self.train_loss_each.reset_states()
    self.reg_loss.reset_states()
    self.total_loss.reset_states()
    self.train_acc.reset_states()
    self.train_acc_each.reset_states()
    self.total_grad.reset_states()

    self.mine_loss.reset_states()


class MFSMetrics:
  def __init__(self):
    self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    self.reg_loss = tf.keras.metrics.Mean(name='regularizer_loss')
    self.total_loss = tf.keras.metrics.Mean(name='total_loss')
    # self.train_acc = tf.keras.metrics.BinaryAccuracy(name='accuracy')
    self.total_grad = tf.keras.metrics.Mean(name='total_grad')

    self.mine_loss = tf.keras.metrics.Mean(name='train_mine_loss')

  def get_metric(self, model, loss, weight_loss, total_loss, norm_grad_all):
    self.train_loss(loss)
    if model.use_regularizer:
      self.reg_loss(weight_loss)
    self.total_loss(total_loss)
    self.total_grad(norm_grad_all)

  def reset_all_states(self):
    self.train_loss.reset_states()
    self.reg_loss.reset_states()
    self.total_loss.reset_states()
    self.total_grad.reset_states()

    self.mine_loss.reset_states()


class RosslerMetrics:
  def __init__(self):
    self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    self.reg_loss = tf.keras.metrics.Mean(name='regularizer_loss')
    self.total_loss = tf.keras.metrics.Mean(name='total_loss')
    # self.train_acc = tf.keras.metrics.BinaryAccuracy(name='accuracy')
    self.total_grad = tf.keras.metrics.Mean(name='total_grad')

    self.mine_loss = tf.keras.metrics.Mean(name='train_mine_loss')

  def get_metric(self, model, loss, weight_loss, total_loss, norm_grad_all):
    self.train_loss(loss)
    if model.use_regularizer:
      self.reg_loss(weight_loss)
    self.total_loss(total_loss)
    self.total_grad(norm_grad_all)

  def reset_all_states(self):
    self.train_loss.reset_states()
    self.reg_loss.reset_states()
    self.total_loss.reset_states()
    self.total_grad.reset_states()

    self.mine_loss.reset_states()


class WeightRecorder:
  '''store weight of rnn models'''

  def __init__(self):

    self.steps = []
    self.W_recs = []
    self.W_recs = []
    self.W_ins = []
    self.W_outs = []
    self.W_fbs = []

  def record(self, model, step):
    self.steps.append(step)
    self.W_recs.append(model.W_rec.numpy())
    self.W_ins.append(model.W_in.numpy())
    self.W_outs.append(model.W_out.numpy())
    self.W_fbs.append(model.W_fb.numpy())

  def restore_weights(self, model, ind=-1):
    ''' restore model weights from ind weight record
      args: 
        model: rnn model
        ind: index of weight record
    '''
    model.W_fb.assign(self.W_fbs[ind])
    model.W_in.assign(self.W_ins[ind])
    model.W_out.assign(self.W_outs[ind])
    model.W_rec.assign(self.W_recs[ind])

# %%

if __name__ == '__main__':
  msepv = MSEPerVariable(n_var=5)
  for i in range(100):
    y_true = tf.zeros(shape=[3,4,5])
    y_pred = tf.random.normal(shape=[3,4,5])
    msepv(y_true, y_pred)
    print(msepv.result())

  msepv.reset_states()
  print(msepv.result()) 


  #%%  

  acpv = BinaryAccuracyPerVariable(n_var=5)
  for i in range(100):
    y_true = tf.zeros(shape=[3,4,5])
    y_pred = tf.random.uniform(shape=[3,4,5])
    acpv(y_true, y_pred)
    print(acpv.result())
# %%
