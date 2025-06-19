# RNN learning with mine


#%%


from numpy import linalg as LA  # linear algebra
import time
from tensorflow.keras import Model
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Layer, Dense, RNN
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'


from mine import MINE, separate_group, run_and_extract_group
from rnnmodel import LeakyRNNModel, build_rnn_model, RandomWithLocalityInitializer, transient
import tasks
from parameters import set_randseed, WMMetrics, MFSMetrics, RosslerMetrics, WeightRecorder
from tools import split_sequence, get_input
from wmtask_train import set_model_for_task, setup_mine, mine_initial_train, train_with_mine, plot_wm_signal

# local files

TASK_NAME = 'wm'
## task setting
DEBUG_MODE = True

#SEED = 2222 #  success with no structural modularity 2023/6/7 wm
SEED = 3333 #  success with no structural modularity 2023/6/7 wm

tf.keras.backend.set_floatx('float32')  # set precision
tf.config.experimental.enable_tensor_float_32_execution(
    False)  # tf32 (混合精度)を使わない
# mser = tf.keras.losses.MeanSquaredError()  # 二乗誤差




def one_trial(seed, gp, debug_mode=True):
  '''
    one trial of model generation and training.
    Note: setting of gp should be done before calling this function.
    args:
      seed: random seed
      gp: global parameters

  '''
  set_randseed(seed)  # for mfs example
  
  # prepare input signal
  task = None
  if gp.task_name == 'wm':
    task = tasks.WorkingMemoryTask(batch_size=gp.bs,
                                  n_mem=gp.n_mem, len_pulse=40, freq_pulse=1.0/500.0, slope=0.1,
                                  transient=gp.transient, time_learn=gp.time_learn, time_test=0, )
  elif gp.task_name == 'mfs':
    task = tasks.MultiFrequencySinusoidalTask(freqs=gp.freqs,
                                              random_phase=True, use_input=True, pred_step=1,
                                              batch_size=gp.bs, transient=gp.transient,
                                              time_learn=gp.time_learn, time_test=0,
                                              )
  elif gp.task_name == 'rossler':
    task = tasks.RosslerPredictionTask(dt=gp.ros_dt,
                                      pred_step=gp.ros_pred_step,
                                      a=gp.ros_params[0], b=gp.ros_params[1], c=gp.ros_params[2],
                                      s_noise=0.0,
                                      batch_size=gp.bs, transient=gp.transient,
                                      time_learn=gp.time_learn, time_test=0,
                                      n_cpu=gp.bs)

  else:
    task = tasks.PeriodicSignalTask(batch_size=gp.bs, transient=0,
                                    time_learn=3600, time_test=3600,
                                    period=1200.0, random_phase=False,
                                    noise_fb=4e-2) #type:ignore
  #split_length = 2000


  # show input and target signals for working memory task
  if debug_mode:
    if task_name == 'wm':
      plot_wm_signal(task)
    elif task_name == 'mfs':
      plt.plot(task.t_transient, task.signals_transient()[0][0, :, :])
    elif task_name == 'rossler':
      plt.plot(task.t_transient, task.signals_transient()[0][0, :, :])

  
  # create model (task-dependent)
  print('Create model!')
  model = set_model_for_task(gp, task)

  # setup MINE network

  mine = setup_mine(model, task, gp)
  print('', flush=True)

  #  setting optimizer
  model_optimizer = tf.keras.optimizers.RMSprop(
      learning_rate=gp.learning_rate, clipnorm=gp.clip_grad)
  mine_optimizer = tf.keras.optimizers.RMSprop(
      learning_rate=gp.learning_rate_mine, clipnorm=gp.clip_grad_mine)

  # print(task_name)
  # setting metrics
  if gp.task_name == 'wm':
    model_metrics = WMMetrics()
  elif gp.task_name == 'mfs':
    model_metrics = MFSMetrics()
  elif gp.task_name == 'rossler':
    model_metrics = RosslerMetrics()
  else:
    model_metrics = None
    
  mi_metric = tf.keras.metrics.Mean(name='mi_metric')
  mi_metric_test = tf.keras.metrics.Mean(name='mi_metric_test')
  w_recorder = WeightRecorder()

  # initial MINE learning phase
  print('Starting initial training of MINE statistic net')
  print('', flush=True)

  #mine_tsf= tf.function(mine_train_step) # mine train step function

  mine_initial_train(model, mine, task, mine_optimizer, gp, mi_metric, mine_tsf=None)

  # save initial model
  save_path = os.path.join(gp.checkpoint_dir, 'ckpt-initial')
  model.save_weights(save_path)
  #TODO: MINEの状態も保存する

  # set of list for recording losses
  losses = {'total': [], 'task': [], 'accs': [],
            'regularizer': [], 'mi_before': [], 'mi_after': []}
  # run training

  losses =  train_with_mine(gp.epochs, n_mine=gp.n_mine_pre, losses=losses, 
                    model=model, 
                    mine=mine, 
                    task=task,
                    gp=gp,                     
                    model_optimizer=model_optimizer,
                    mine_optimizer=mine_optimizer,
                    w_recorder=w_recorder,
                    model_metrics=model_metrics, 
                    mi_metric=mi_metric,
                    debug_mode=debug_mode)

  print('-----------------')
  print('training done')
  print('-----------------')
  # save losses
  if gp.task_name == 'wm':
    data_dir = './data/wm'
  elif gp.task_name == 'mfs':
    data_dir = './data/mfs'
  elif gp.task_name == 'rossler':
    data_dir = './data/rossler'
  else:
    data_dir = './data/periodic'

  os.makedirs(data_dir, exist_ok=True)

  with open(os.path.join(data_dir, 'losses.pkl'), 'wb') as f:
    pickle.dump(losses, f)
  # save weights
  with open(os.path.join(data_dir, 'w_recorder.pkl'), 'wb') as f:
    pickle.dump(w_recorder, f)
  return model, task, losses, w_recorder

  # codes for making figures are moved to wm_analysis_essence.py
  


#%%


if __name__ == '__main__':
  
  # GPU memory を使いすぎない設定 (一部のマシンではこれをいれないでKeras使うとエラーになることもある)
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
      tf.config.experimental.set_memory_growth(physical_devices[k], True)
      print('memory growth:', tf.config.experimental.get_memory_growth(
          physical_devices[k]))
  else:
    print("Not enough GPU hardware devices available")

  #matplotlib オフラインーオンライン切り替え関連
  import matplotlib as mpl

  def is_env_notebook():
    """Determine whether is the environment Jupyter Notebook"""
    if 'get_ipython' not in globals():
        # Python shell
        return False
    from IPython import get_ipython # type:ignore
    env_name = get_ipython().__class__.__name__
    if env_name == 'TerminalInteractiveShell':
        # IPython shell
        return False
    # Jupyter Notebook
    return True

  if not is_env_notebook():
    print('use AGG')
    mpl.use('Agg')  # off-screen use of matplotlib

  #%%
  task_name = TASK_NAME
  
  if task_name == 'wm':
    import gp_wm as gp
  elif task_name == 'mfs':
    import gp_mfs as gp
  elif task_name == 'rossler':
    import gp_rossler as gp
  else:
    print('Error: task_name is not correct')
    exit(1)
    
  losses = one_trial(seed=SEED, gp=gp, debug_mode=DEBUG_MODE)

# %%
