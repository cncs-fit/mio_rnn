# -*- coding: utf-8 -*-
# Define timeseries tasks for MI-RNN project

#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing
from typing import Optional, Union, List, Tuple, Dict, Callable, Any

#functions for making teacher signal


def periodic_signal(length, freq=1.0/1200.0, amp=1.3, init_phase=0, batch_size=1):
  '''makes periodic teacher signal
  args:
      tp: TimeParam object
      freq: base frequency
      amp: amplitude
      init_phase: initial phase: (batch) dim array or scalar
      batch_size: batch_size
  returns:
   ft:  (batch, length, 1) dim signals
  '''
  ts = np.arange(length).reshape((1, length))

  if np.isscalar(init_phase):
    ph0 = init_phase * np.ones(shape=(batch_size, 1))
  else:
    ph0 = init_phase.reshape((batch_size, 1))

  ft = (amp/1.0)*np.sin(1.0*(2*np.pi*freq*ts + ph0)) + \
      (amp/2.0)*np.sin(2.0*(2*np.pi*freq*ts + ph0)) + \
      (amp/6.0)*np.sin(3.0*(2*np.pi*freq*ts + ph0)) + \
      (amp/3.0)*np.sin(4.0*(2*np.pi*freq*ts + ph0))

  ft = ft/1.5

  return np.reshape(ft, (batch_size, length, 1))


def sinusoidal_signal(length, freq=1.0/1200.0, amp=1.3):
  '''makes sinusoidal teacher signal
  args:
      freq: base frequency
      amp: amplitude
  '''
  ft = (amp/1.0)*np.sin(2.0*np.pi*freq*np.arange(length))
  return np.reshape(ft, [length, 1])


def sawtooth_signal(length, freq=1.0/1200.0, amp=1.0, init_phase=0, batch_size=1):
  '''makes triangular wave for teacher signal
  args:
      tp: TimeParam object
      freq: base frequency
      amp: amplitude
  '''
  T = 1.0/freq  # period
  slope = 4.0*amp/T

  if np.isscalar(init_phase):
    ph0 = init_phase * np.ones(shape=(batch_size, 1))
  else:
    ph0 = np.array(init_phase).reshape((batch_size, 1))

  t_peri = (ph0+np.arange(length, dtype=np.float).reshape(1, length)) % T

  phase = 4*t_peri // T

  ft = slope*(t_peri*(phase == 0)
              - (t_peri - T/2)*((1 <= phase) & (phase <= 2))
              + (t_peri - T)*(phase == 3))
  return np.reshape(ft, [batch_size, length, 1])


class Task:
  ''' Task情報を提供するクラス
     時間の情報，入力信号，教師信号を管理
     transient, training, test区間と続く．
  '''

  def __init__(self, period=1200.0,
               batch_size=1, transient=1000, time_learn=5000, time_test=5000,
               N_in=1):
    self.N_in = N_in
    self.batch_size = batch_size
    self.transient = transient
    self.time_learn = time_learn
    self.time_test = time_test
    self.total_time = transient + time_learn + time_test
    self.tstart_training = transient
    self.end_training = transient + time_learn



    self.t_total = np.arange(0, self.total_time+1)
    self.t_transient = np.arange(0, transient)
    self.t_learn = np.arange(transient, transient+time_learn)
    self.t_test = np.arange(transient+time_learn,
                            transient+time_learn+time_test+1)
    self.input_signal = np.zeros(
        (self.batch_size, self.total_time+1, self.N_in), dtype=float)  # dummy

    self.target_signal = np.zeros(
        (self.batch_size, self.total_time+1, 1))  # dummy
    self.fb_signal = np.zeros(
        (self.batch_size, self.total_time+1, self.N_in), dtype=float)  # dummy


  def gen_signals():
    ''' generate input and target signal'''
    pass
  
  # それぞれの区間の入力，教師信号を返す関数
  def signals_transient(self, fb=False)->Tuple:
    if fb:
      return (self.input_signal[:, self.t_transient, :], self.target_signal[:, self.t_transient, :],
              self.fb_signal[:, self.t_transient, :])
    else:
      return (self.input_signal[:, self.t_transient, :], self.target_signal[:, self.t_transient, :])

  def signals_learning(self, fb=False)->Tuple:
    if fb:
      return (self.input_signal[:, self.t_learn, :], self.target_signal[:, self.t_learn, :],
              self.fb_signal[:, self.t_learn, :])
    else:
      return (self.input_signal[:, self.t_learn, :], self.target_signal[:, self.t_learn, :])

  def signals_test(self, fb=False):
    if fb:
      return (self.input_signal[:, self.t_test, :], self.target_signal[:, self.t_test, :],
              self.fb_signal[:, self.t_test, :])
    else:
      return (self.input_signal[:, self.t_test, :], self.target_signal[:, self.t_test, :])


class PeriodicSignalTask(Task):
  ''' 周期関数を学習するタスク
  
  '''

  def __init__(self, period=1200.0,
               batch_size=1, transient=1000,
               time_learn=5000, time_test=5000,
               N_in=1, random_phase=True,
               noise_fb=0):
    super().__init__(batch_size=batch_size, transient=transient, time_learn=time_learn,
                     time_test=time_test, N_in=N_in)

    self.period = period
    self.random_phase = random_phase  # 初期位相をランダム
    self.noise_fb = noise_fb
    self.gen_signals()

  def gen_signals(self):
    ''' generate input and target signal'''
    if self.random_phase:
      ph0 = 2*np.pi * np.random.uniform(size=(self.batch_size))
    else:
      # ph0 = 2*np.pi * np.arange(self.batch_size)/self.batch_size
        ph0 = 0

    # print(ph0)
    # make input signal
    self.input_signal = np.zeros(
        (self.batch_size, self.total_time+1, self.N_in))  # do not use
    # make target_signal
    self.signal = periodic_signal(self.total_time+2, freq=1/self.period,
                                  init_phase=ph0, batch_size=self.batch_size,
                                  amp=1.0)

    self.target_signal = self.signal[:, 1:, :].copy()
    self.fb_signal = self.signal[:, 0:-1, :].copy()
    self.fb_signal += self.noise_fb * np.random.randn(*(self.fb_signal.shape))


class MultiFrequencySinusoidalTask(Task):
  ''' multi-frequency sinusoidal wave prediction task.
      異なった周波数の正弦波を同時に生成するタスク
      位相差はサンプルごとにランダムに変更することができる．

  '''

  def __init__(self, freqs=[np.sqrt(2)/1000, np.sqrt(3)/1000],
               random_phase=True, use_input=True, pred_step=1,
               s_noise=0.0,
               batch_size=1, transient=1000,
               time_learn=5000, time_test=0,
               ):
    '''
      freqs: list of frequency. its size will be N_in
      random_phase: randomize initial phase of each wave
      use_input: if true, sinusoidal is fed into input and the task is predict future step,
          if false, no input was given and the task is to generate sinusoidal
      pred_step: indicate how many steps ahead to predict.
      s_noise: standard deviation of noise applied to input signal
      batch_size: batch_size
      transient: transient length
      time_learn, time_test: length for learning and testing
    '''
    N_in = len(freqs)
    super().__init__(batch_size=batch_size, transient=transient, time_learn=time_learn,
                     time_test=time_test, N_in=N_in)

    self.freqs = np.array(freqs)
    self.random_phase = random_phase  # 初期位相をランダム
    self.use_input = use_input  # インプットに信号を入れる．
    self.pred_step = pred_step  # 入力の何ステップ先を予測するか．
    self.s_noise = s_noise
    self.use_noise = (s_noise == 0.0)

    self.gen_signals()

  def gen_signals(self):
    ''' generate input and target signal'''
    if self.random_phase:
      ph0 = 2*np.pi * np.random.uniform(size=(self.batch_size, self.N_in))
    else:
        ph0 = np.zeros(shape=(self.batch_size, self.N_in))

    # print(ph0)
    # make input signal
    # make target_signal

    #各時刻の位相，周波数ごとにつくる．(ブロードキャスト使うのでreshapeしている．) 出力サイズは (timestep, N_in)
    phi_t = 2.0 * np.pi * np.arange(1+self.total_time+self.pred_step).reshape(
        1+self.total_time+self.pred_step, 1)*self.freqs.reshape(1, 2)
    # 初期位相ぶんずらす（ブロードキャスト使う） (batch_size, timestep, N_in)
    phi_t_batch = ph0.reshape(self.batch_size, 1, self.N_in) + phi_t
    self.signal = np.sin(phi_t_batch)

    if self.use_input:
      self.input_signal = self.signal[:, 0:(self.total_time+1), :].copy()
      self.input_signal += self.s_noise * \
          np.random.normal(size=self.input_signal.shape)
    else:
      self.input_signal = np.zeros(
          (self.batch_size, self.total_time+1, self.N_in))  # do not use

    self.target_signal = self.signal[:, self.pred_step:(
        self.pred_step+self.total_time+1), :].copy()
    self.fb_signal = self.signal[:, self.pred_step:(
        self.pred_step+self.total_time+1), :].copy()
    #self.fb_signal += self.noise_fb * np.random.randn( *(self.fb_signal.shape) )


class WorkingMemoryTask(Task):
  ''' WorkingMemoryタスクの信号を作るクラス
  
  '''

  def __init__(self, n_mem=2, len_pulse=50, freq_pulse=1.0/1000.0, slope=0.2,
               batch_size=1, transient=1000, time_learn=5000, time_test=5000,
               single_line=False, add_task=None,
               ):
    '''
    args:
     n_mem: number of working memory. N_in and N_out has 2*n_mem dimension
     len_pulse: duration of pulse 
    freq_pulse: frequency (or probability) of pulse  10* dt *  1/1000.0 #
    single_line: if true, using single input signal. if false, using dual input lines, indicating on and off.

    '''
    if single_line:
      self.N_in = n_mem
    else:
      self.N_in = n_mem*2
    super().__init__(batch_size=batch_size, transient=transient, time_learn=time_learn,
                     time_test=time_test, N_in=self.N_in)

    #入力信号のパラメータ
    self.n_mem = n_mem
    self.len_pulse = len_pulse
    self.freq_pulse = freq_pulse
    self.slope = slope
    self.single_line = single_line
    self.N_out = n_mem
    self.add_task = add_task # and_, or_, or 

    if self.add_task is not None:
      self.N_out=3
    self.gen_signals()
    # self.N_in = n_mem


  def gen_signals(self):
    if self.single_line:
      pass #TODO: 未実装
      # self.gen_signals_single_input()
    else:
      self.gen_signals_dual_input()

  def gen_signals_dual_input(self):
    ''' generate input and target signal'''
    # make input signal
    self.input_signal = np.zeros(
        (self.batch_size, self.total_time+1, self.n_mem*2), dtype=float)  # 入力信号
    #flag of pulse pulseが立ち上がるときに１，ほかでは0
    self.v_start = 1.0*(np.random.rand(self.batch_size,
                                       self.total_time+2, self.n_mem*2) < self.freq_pulse)
    #v_tag=  np.zeros( (tp.tmax_step+1,n_mem*2))#信号開始からlen_commandで1 矩形波．
    self.list_pulse_start = []  # それぞれの開始時刻のリスト  [batch][mem_index][order]
    for b in range(self.batch_size):
      l_b = []
      for i in range(2*self.n_mem):
        # l =[t for t in range(tp.tmax_step) if v_start[t,i] ==1]#開始時刻リスト
        l, = np.where(self.v_start[b, :, i] == 1)
        l_b.append(l)
        for tp_start in l:
          self.input_signal[b, tp_start:min(
              tp_start+self.len_pulse, self.total_time+2), i] = 1.0
      self.list_pulse_start.append(l_b)

    # target signal (teacher)
    self.signal = np.zeros(
        (self.batch_size, self.total_time+2, self.n_mem), dtype=float)
    # self.signal[:, 0, :] = 2.0*(self.v_start[:, 0, 0::2]-0.5)
    self.state = np.zeros(
        (self.batch_size, self.total_time+2, self.n_mem), dtype=float)
    self.state[:, 0, :] = 2.0*(self.v_start[:, 0, 0::2]-0.5)

    #stateの式の作り方．
    # １項： on がきたら 1
    # 2項: on がきていなくoffがきていたら -1
    # 3項: どちらもきていなかったら，前と同じ
    for t in range(1, self.total_time+1):
      self.state[:, t, :] = self.v_start[:, t, 0:2*self.n_mem:2] +  \
          -1*(self.v_start[:, t, 0:2*self.n_mem:2] == 0)*self.v_start[:, t, 1:2*self.n_mem:2] +\
          (self.v_start[:, t, 0:2*self.n_mem:2] == 0)*(self.v_start[:,
                                                                    t, 1:2*self.n_mem:2] == 0)*self.state[:, t-1, :]

      self.signal[:, t, :] = np.maximum(-1.0, np.minimum(
          1.0, self.signal[:, t-1, :] + self.slope*self.state[:, t, :]))

    self.target_signal = self.signal[:, 1:, :]
    self.fb_signal = self.signal[:, 0:-1, :]

    if self.add_task is not None:
      self.on_off_signal = self.target_signal.copy() # preserve previous target signal 
      if self.add_task == 'xor_':
        self.xor_signal = np.expand_dims(- self.target_signal[:, :, 0] * self.target_signal[:,:,1], axis=2) # xor
        self.target_signal = np.concatenate([self.on_off_signal, self.xor_signal], axis=2)
      elif self.add_task == 'or_':
        self.or_signal = np.expand_dims(np.maximum(self.target_signal[:, :, 0], self.target_signal[:,:,1]), axis=2) # or       
        self.target_signal = np.concatenate([self.on_off_signal, self.or_signal], axis=2)
      elif self.add_task == 'and_':
        self.and_signal = np.expand_dims(np.minimum(self.target_signal[:, :, 0], self.target_signal[:,:,1]), axis=2) 
        self.target_signal = np.concatenate([self.on_off_signal, self.and_signal], axis=2)
      
      # self.fb_signal = np.expand_dims(- self.fb_signal[:, :, 0] * self.fb_signal[:,:,1],axis=2)


  # def gen_signals_single_input(self):
  #   ''' generate input and target signal
  #       これは現在動作チェックしていない．
  #   '''
  #   # make input signal
  #   self.input_signal = np.zeros(
  #       (self.batch_size, self.total_time+1, self.n_mem), dtype=float)  # 入力信号
  #   #flag of pulse pulseが立ち上がるときに１，ほかでは0
  #   self.v_start_on = 1.0*(np.random.rand(self.batch_size,
  #                                         self.total_time+2, self.n_mem) < self.freq_pulse)
  #   self.v_start_off = 1.0*(np.random.rand(self.batch_size,
  #                                          self.total_time+2, self.n_mem) < self.freq_pulse)

  #   #v_tag=  np.zeros( (tp.tmax_step+1,n_mem*2))#信号開始からlen_commandで1 矩形波．
  #   self.list_pulse_start = []  # それぞれの開始時刻のリスト  [batch][mem_index][order]
  #   for b in range(self.batch_size):
  #     l_b = []
  #     for i in range(self.n_mem):
  #       # l =[t for t in range(tp.tmax_step) if v_start[t,i] ==1]#開始時刻リスト
  #       l_on, = np.where(self.v_start_on[b, :, i] == 1)
  #       l_off, = np.where(self.v_start_off[b, :, i] == 1)
  #       l_b.append(l)
  #       for tp_start in l:
  #         self.input_signal[b, tp_start:min(
  #             tp_start+self.len_pulse, self.total_time+2), i] = 1.0
  #     self.list_pulse_start.append(l_b)

  #   # target signal (teacher)
  #   self.signal = np.zeros(
  #       (self.batch_size, self.total_time+2, self.n_mem), dtype=float)
  #   # self.signal[:, 0, :] = 2.0*(self.v_start[:, 0, 0::2]-0.5)
  #   self.state = np.zeros(
  #       (self.batch_size, self.total_time+2, self.n_mem), dtype=float)
  #   self.state[:, 0, :] = 2.0*(self.v_start[:, 0, 0::2]-0.5)

  #   #stateの式の作り方．
  #   # １項： on がきたら 1
  #   # 2項: on がきていなくoffがきていたら -1
  #   # 3項: どちらもきていなかったら，前と同じ
  #   for t in range(1, self.total_time+1):
  #     self.state[:, t, :] = self.v_start[:, t, 0:2*self.n_mem:2] +  \
  #         -1*(self.v_start[:, t, 0:2*self.n_mem:2] == 0)*self.v_start[:, t, 1:2*self.n_mem:2] +\
  #         (self.v_start[:, t, 0:2*self.n_mem:2] == 0)*(self.v_start[:,
  #                                                                   t, 1:2*self.n_mem:2] == 0)*self.state[:, t-1, :]

  #     self.signal[:, t, :] = np.maximum(-1.0, np.minimum(
  #         1.0, self.signal[:, t-1, :] + self.slope*self.state[:, t, :]))

  #   self.target_signal = self.signal[:, 1:, :]
  #   self.fb_signal = self.signal[:, 0:-1, :]


class SawToothSignalTask(Task):

  def __init__(self, period=1200.0,
               batch_size=1, transient=1000, time_learn=5000, time_test=5000,
               N_in=1):
    super().__init__(batch_size=batch_size, transient=transient, time_learn=time_learn,
                     time_test=time_test, N_in=N_in)

    self.period = period
    self.rand_phase = True  # 初期位相をランダム
    self.gen_signals()

  def gen_signals(self):
    ''' generate input and target signal'''
    if self.rand_phase:
      ph0 = self.period * np.random.uniform(size=(self.batch_size))
    else:
      ph0 = self.period * np.arange(self.batch_size)/self.batch_size
    #print(ph0)
    # make input signal
    self.input_signal = np.zeros(
        (self.batch_size, self.total_time+1, self.N_in))  # do not use
    # make target_signal
    self.target_signal = sawtooth_signal(self.total_time, freq=1/self.period,
                                         batch_size=self.batch_size, init_phase=ph0,
                                         amp=1.0)


class Rossler:
  """Rossler equation class"""

  def __init__(self, a, b, c):
    """
    rossler equation
    """
    self.a = a
    self.b = b
    self.c = c

  def __call__(self, t, x):

    dxdt = -x[1]-x[2]
    dydt = x[0] + self.a * x[1]
    dzdt = self.b + x[0] * x[2] - self.c * x[2]
    return [dxdt, dydt, dzdt]


class RosslerPredictionTask(Task):
  ''' Prediction of Rossler system.

  '''

  def __init__(self,
               dt=0.1,
               pred_step=1,
               a=0.2, b=2, c=5.7,
               s_noise=0.0,
               batch_size=20, transient=1000,
               time_learn=5000, time_test=0,
               n_cpu=None
               ):
    '''
      dt: timestep
      pred_step: How many steps forward to predict
      a,b,c: parameters
      s_noise: standard deviation of noise applied to input signal
      batch_size: batch_size
      transient: transient length
      time_learn, time_test: length for learning and testing
    '''
    N_in = 3
    self.N_out = 3
    super().__init__(batch_size=batch_size, transient=transient, time_learn=time_learn,
                     time_test=time_test, N_in=N_in)

    self.dt = dt
    self.pred_step = pred_step  # 入力の何ステップ先を予測するか．
    self.a = a
    self.b = b
    self.c = c
    self.s_noise = s_noise
    self.use_noise = (s_noise == 0.0)
    self.ro = Rossler(a=0.2, b=0.2, c=5.7)
    self.t_start_ro = 0
    self.t_end_ro = self.total_time*self.dt
    if n_cpu == None:
      self.n_cpu = multiprocessing.cpu_count()
    else:
      self.n_cpu = n_cpu
    
    global pool_ros 
    pool_ros = Pool(self.n_cpu) # ここで並列計算用プロセスを作っておく

    self.ts = np.arange(self.t_start_ro, self.t_end_ro +
                        (1+self.pred_step)*dt, dt)  # 予測ステップ分だけ多く生成

    self.gen_signals()
  
  # def __del__(self):
  #   pool_ros.close()
  #   super().__del__()

  def solv(self, x0_n):
      """ wrapper function"""
      return solve_ivp(self.ro, t_span=[self.ts[0], self.ts[-1]], y0=x0_n, t_eval=self.ts)['y'].transpose()

  def gen_signals(self):
    ''' generate input and target signal'''

    # set initial conditions
    x0 = np.random.rand(self.batch_size, 3)
    x0[:, 0] = 5.0*(2*x0[:, 0]-1.0)
    x0[:, 1] = 5.0*(2.0 * x0[:, 1] - 1.0)
    x0[:, 2] = 0.2*x0[:, 2]

    # with Pool(self.n_cpu) as pool:
    Xs = pool_ros.map(self.solv, x0)
    # Xs = [ solve_ivp(self.ro, t_span= [self.ts[0],self.ts[-1]], y0=x0_n,t_eval=self.ts)['y']
    # Xs = [ solve_ivp(self.ro, t_span= [self.ts[0],self.ts[-1]], y0=x0_n,t_eval=self.ts)['y'] for x0_n in x0 ]

    self.signal = np.array(Xs)/np.array([5., 5., 10.])
    self.input_signal = self.signal[:, 0:(self.total_time+1), :].copy()
    self.target_signal = self.signal[:, self.pred_step:(
        self.pred_step+self.total_time+1), :].copy()
    self.fb_signal = self.signal[:, self.pred_step:(
        self.pred_step+self.total_time+1), :].copy()
    if self.use_noise:
      self.input_signal += self.s_noise * \
          np.random.normal(size=self.input_signal.shape)


#%%
if __name__ == '__main__':
  pass
  import matplotlib.pyplot as plt

  wm_task = WorkingMemoryTask(n_mem=2, len_pulse=50, 
                              freq_pulse=1.0/1000.0, slope=0.2,
                              batch_size=1, transient=1000, 
                              time_learn=5000, time_test=5000,
                              add_task = None                              
                              )

  ps_task = PeriodicSignalTask(period=1200.0,
                               batch_size=1, transient=1000,
                               time_learn=5000, time_test=0,
                               N_in=1, random_phase=True,
                               noise_fb=0)
  inputs_learn, target_learn = ps_task.signals_learning()
  plt.plot(target_learn[0, :, :])

  mfs_task = MultiFrequencySinusoidalTask(freqs=[2.0*np.sqrt(2)/1000, 3.0*np.sqrt(3)/1000],
                                          random_phase=True, use_input=True, pred_step=1,
                                          batch_size=10, transient=1000,
                                          time_learn=2000, time_test=0,
                                          )
  inp_tra, tar_tra = mfs_task.signals_transient()
  inp_learn, tar_learn = mfs_task.signals_learning()
  plt.plot(mfs_task.t_transient, inp_tra[0, :, :], '.-')
  plt.plot(mfs_task.t_learn, inp_learn[0, :, :], '-')

  plt.plot(range(mfs_task.transient), inp_tra[3, :, :], '.-')
  plt.show()

  l_on = np.array([10, 20, 30])
  l_off = np.array([15, 40, 50])

  pmone = np.concatenate([np.ones_like(l_on), -1 * np.ones_like(l_off)])
  l_two = np.concatenate([l_on, l_off])
  sorted_index = np.argsort(l_two)
  print(pmone)
  print(l_two)
  print(sorted_index)

  on_or_off = pmone[sorted_index]
  print(on_or_off)



# %%
  ro = Rossler(a=0.2, b=0.2, c=5.7)
  x0 = np.array([5, 0, 0], dtype=np.float64)

  dt = 0.05
  tstart = 0
  tend = 100
  ts = np.arange(tstart, tend+dt, dt)
  sol_rossler = solve_ivp(ro, t_span=[ts[0], ts[-1]], y0=x0, t_eval=ts)
  X = sol_rossler['y']
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)

  ax.plot(ts, X[0], '.-')

  #%% Rossler
  # generate task object
  task_ro = RosslerPredictionTask(dt=0.1,
                                  pred_step=1,
                                  a=0.2, b=2, c=5.7,
                                  s_noise=0.0,
                                  batch_size=50, transient=1000,
                                  time_learn=4000, time_test=0,)
  input_transient, target_transient = task_ro.signals_transient()
  inputs, target = task_ro.signals_learning()
  # %%
  plt.plot(task_ro.t_transient, input_transient[0], task_ro.t_learn, inputs[0])

  # %%
  plt.plot(inputs[0, :, 0].T, inputs[0, :, 1].T)
  # %%
  plt.plot(task_ro.t_learn, inputs[0], task_ro.t_learn, target[0], '.-')
  # %% speed test

  for e in range(100):
    task_ro.gen_signals()
    input_transient, target_transient = task_ro.signals_transient()
    inputs, target = task_ro.signals_learning()
    m_i = inputs.mean(axis=(0, 1))
    print(f'{e}, {m_i}')

# %%
