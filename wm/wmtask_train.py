# RNN learning with mine
#TODO: recurrent initializer の設定をみなおす

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

# local files

#import gp_wm as gp
from mine import MINE, separate_group, run_and_extract_group
from rnnmodel import LeakyRNNModel, build_rnn_model, RandomWithLocalityInitializer, transient
import tasks
from parameters import set_randseed, WMMetrics, MFSMetrics, RosslerMetrics, WeightRecorder
from tools import split_sequence, get_input



TASK_NAME = 'wm'
## task setting
DEBUG_MODE = True
# SEED = 1110 # success 2022/3/4 mfs
# SEED = 1115  # success 2022/3/4 mfs

#SEED = 2222 #  success with no structural modularity 2023/6/7 wm
SEED = 3333 #  success with a structural modularity 2023/6/7 wm

tf.keras.backend.set_floatx('float32')  # set precision
tf.config.experimental.enable_tensor_float_32_execution(
    False)  # tf32 (混合精度)を使わない
mser = tf.keras.losses.MeanSquaredError()  # 二乗誤差




@tf.function
def train_step(model, inputs, teacher,
               model_metrics,
               optimizer,
               initial_state=None,
               ):
  '''training step for RNN model (without MINE)
  '''
  with tf.GradientTape() as tape:
    z_tr = model(inputs, initial_state=initial_state)
    loss = mser(teacher, z_tr)
    if model.use_regularizer:
      weight_loss = tf.add_n(model.losses)
      total_loss = loss + weight_loss
    else:
      total_loss = loss

  gradients = tape.gradient(total_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  norm_grad_all = norm_grad(gradients)
  if gp.task_name == 'wm':  # task_name is a global variable
    model_metrics.get_metric(model, loss, weight_loss,
                             total_loss, teacher, z_tr, norm_grad_all)
  elif gp.task_name == 'mfs':
    model_metrics.get_metric(model, loss,
                             weight_loss, total_loss, norm_grad_all)
  elif gp.task_name == 'rossler':
    model_metrics.get_metric(model, loss,
                             weight_loss, total_loss, norm_grad_all)

  return z_tr, model.r, model.last_state


# def get_wm_metric(model, model_metrics, loss, weight_loss, total_loss, teacher, z_tr, norm_grad_all):
#   model_metrics.train_loss(loss)
#   if model.use_regularizer:
#     model_metrics.reg_loss(weight_loss)
#   model_metrics.total_loss(total_loss)
#   model_metrics.train_acc(tf.round((teacher+1.0)/2.0),
#                           (z_tr+1.0)/2.0)  # metrics にいれている
#   model_metrics.total_grad(norm_grad_all)


# def get_mfs_metric(model, model_metrics, loss, weight_loss, total_loss, norm_grad_all):
#   model_metrics.train_loss(loss)
#   if model.use_regularizer:
#     model_metrics.reg_loss(weight_loss)
#   model_metrics.total_loss(total_loss)
#   model_metrics.total_grad(norm_grad_all)


##@tf.function

def model_train_step_with_mine(model, mine, inputs, teacher,
                               model_optimizer,
                               gp,
                               model_metrics=None, metric_mi=None,
                               initial_state=None,
                               use_mi=True,
                               ):
  '''通常のロス，正則化ロスに加え，MIの最大化/最小化を学習する
  args:
    model: RNNModel
    mine: MINE network
    inputs: input signal
    teacher: target signal
    model_optimizer: optimizer for RNN model
    gp: global parameters
    model_metrics: custom metrics class object for model
    metric_mi: custom metrics class object for MI
    initial_state: initial state of RNN
    use_mi: if True, MI is used as a loss function
  '''

  with tf.GradientTape() as tape:
    z_tr, r1, r2 = run_and_extract_group(
        model, inputs, initial_state=initial_state, gp=gp)
    mi = mine(r1, r2)  # estimation of MI by MINE
    loss = mser(teacher, z_tr)  # mean squared error for teacher and output
    if model.use_regularizer:
      # model.losses include regularizer loss?
      weight_loss = tf.add_n(model.losses)
      total_loss = loss + weight_loss
    else:
      total_loss = loss

    if use_mi:
      total_loss += gp.lambda_mi * mi

  gradients = tape.gradient(total_loss, model.trainable_variables)
  model_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  norm_grad_all = norm_grad(gradients)
  if gp.task_name == 'wm':  # task_name is a global variable
    model_metrics.get_metric(model, loss, weight_loss,
                             total_loss, teacher, z_tr, norm_grad_all)
  elif gp.task_name == 'mfs':
    model_metrics.get_metric(model, loss,
                             weight_loss, total_loss, norm_grad_all)
  elif gp.task_name == 'rossler':
    model_metrics.get_metric(model, loss,
                             weight_loss, total_loss, norm_grad_all)

  if use_mi:
    model_metrics.mine_loss(mi)
  # model_metrics.train_loss(loss)
  # if model.use_regularizer:
  #   model_metrics.reg_loss(weight_loss)
  # model_metrics.total_loss(total_loss)
  # model_metrics.train_acc(tf.round((teacher+1)/2), (z_tr+1)/2)  # metrics にいれている

  # norm_grad_all = norm_grad(gradients)
  # model_metrics.total_grad(norm_grad_all)

  if metric_mi is not None:
    metric_mi(mi)

  return z_tr, model.r, model.last_state


def model_train_epoch_with_mine(model, mine, task,
                                model_optimizer,
                                gp,
                                model_metrics=None, metric_mi=None,
                                use_mi=True,
                                model_tsf=None,
                                ):
  ''' one-epoch training of RNN model with MINE
  '''
  if model_tsf is None:
    model_tsf = tf.function(model_train_step_with_mine)

  # generate signals and split
  input_signal_stack, teacher_signal_stack, inputs_transient, teacher_transient = get_input(
      task, gp)
  n_batch = len(teacher_signal_stack)  # バッチ数の確認

  # initial state
  #x0_zero = tf.zeros(shape=[gp.bs, gp.N], dtype=gp.mydtype)
  x0 = model.noise_g.normal(shape=[gp.bs, gp.N], dtype=gp.mydtype)  # 初期値

  initial_state = x0
  # transient run
  if task.transient > 0:
    _, _, initial_state = transient(model, inputs_transient, initial_state)
  # loop for batch
  for i, (inp, tea) in enumerate(zip(input_signal_stack,
                                     teacher_signal_stack,
                                     )):

    z, r, last_state = model_tsf(model, mine, inp, tea,
                                                  model_optimizer, gp, model_metrics=model_metrics, metric_mi=metric_mi,
                                                  initial_state=initial_state, use_mi=use_mi)  # ここで訓練１バッチ回す
    initial_state = last_state
    if i % 1 == 0:
      print('.', end='')


## @tf.function

def mine_train_step(model, mine, inputs, initial_state=None,
                    metric_mi=None,
                    mine_optimizer=None,
                    gp=None,
                    ):
  ''' one-training of  statistic network
  args:
    model: RNNModel
    mine: MINE network
    inputs: input signal
    initial_state: initial state of RNN
    metric_mi: custom metrics class object for MI
    mine_optimizer: optimizer for MINE
    gp: global parameters
  '''

  z, r1, r2 = run_and_extract_group(
      model, inputs, initial_state=initial_state, gp=gp)
  with tf.GradientTape() as tape:
    mi_objective = -mine(r1, r2)

  gradients = tape.gradient(mi_objective, mine.trainable_variables)
  mine_optimizer.apply_gradients(zip(gradients, mine.trainable_variables))
  if metric_mi is not None:
    metric_mi(-mi_objective)
  return z, model.x, model.last_state


def mine_train_epoch(model, mine, task, initial_state=None,
                     mine_optimizer=None,
                     gp=None, metric_mi=None,
                      mine_tsf=None, ):
  ''' train statistic network 1 epoch'''
  # task.gen_signals()  # 信号生成
  # inputs_transient, _ = task.signals_transient()
  # input_signal, teacher_signal = task.signals_learning()  # 入力と教師
  # input_signal_stack = split_sequence(input_signal, gp.split_length)  # 分割
  # teacher_signal_stack = split_sequence(teacher_signal, gp.split_length) # mineではteacherは使わない

  if mine_tsf is None:
    mine_tsf= tf.function(mine_train_step)
  input_signal_stack, teacher_signal_stack, inputs_transient, teacher_transient = get_input(
      task, gp)

  if gp.transient > 0:
    _, _, initial_state = transient(model, inputs_transient, initial_state)

  for input_sp in input_signal_stack:
    z, X, x_last = mine_tsf(model, mine, input_sp, initial_state,
                                   metric_mi=metric_mi,
                                   mine_optimizer=mine_optimizer,
                                   gp=gp,
                                   )
    initial_state = x_last


def mine_initial_train(model, mine, task, mine_optimizer, gp, mi_metric, mine_tsf=None):
  ''' initial training of MINE before RNN training
  args:
    model: RNNModel
    mine: MINE network
    task: task object
    mine_optimizer: optimizer for MINE
    gp: global parameters
  '''
  if mine_tsf==None:
    mine_tsf= tf.function(mine_train_step)

  for n in range(gp.initial_mine_epoch):
    x0 = tf.random.normal(shape=[gp.bs, gp.N], dtype=gp.mydtype)

    mine_train_epoch(model, mine, task=task,
                     initial_state=x0, mine_optimizer=mine_optimizer, metric_mi=mi_metric, gp=gp, mine_tsf=mine_tsf)
    print(
        f'\r initial statistic net training: {n} {mi_metric.result().numpy()}', end='')
    # print('', flush=True)

  return



def norm_grad(gradients):
  ''' return total norm of the parameters
  '''
  square_gs = [tf.math.square(g) for g in gradients]
  square_norm_grads = [tf.math.reduce_sum(sg) for sg in square_gs]
  norm_all_grad = tf.sqrt(tf.reduce_sum(square_norm_grads))
  return norm_all_grad


def plot_wm_signal(task, length=5000):
  input_signal, teacher_signal, fb_signal = task.signals_learning(fb=True)
  n_ax = 3 * task.n_mem

  fig, axes = plt.subplots(n_ax, 1)
  for m in range(task.n_mem):
    axes[3*m].plot(input_signal[0, 0:length, 2*m], '-k')
    axes[3*m+1].plot(input_signal[0, 0:length, 2*m+1], '-k')
    axes[3*m+2].plot(teacher_signal[0, 0:length, m])


def set_model_for_task(gp, task):
  ''' create model and build it with task-dependent input signal
  '''
  model = build_rnn_model(gp)

  # generate input and feed it to the model to initialize the model
  # stack になっているのは，長い時系列を分割して使うときのため．現在は，1つの要素しかない．
  inputs_stack, target_stack, inputs_transient, target_transient = get_input(
      task, gp)

  # random initial state?
  x0 = tf.random.normal(shape=[gp.bs, gp.N], dtype=gp.mydtype)
  z = model(inputs_stack[0], initial_state=x0)  # test run
  model.additional_setup()  # set convenient name for important weights
  model.summary()
  return model


def setup_mine(model, task, gp):
  ''' setup MINE network
  
  args:
    model: RNNModel
    task: task object
    gp: global parameters
    returns:
      mine: MINE network
  '''

  mine = MINE(n_hidden=gp.N_mt_hidden, s_noise=1e-1, mydtype=gp.mydtype)
  inputs_stack, _, _, _ = get_input(
      task, gp)

  x0 = tf.random.normal(shape=[gp.bs, gp.N], dtype=gp.mydtype)
  z, r1, r2 = run_and_extract_group(model, inputs_stack[0], x0, gp)
  out_stat = mine(r1, r2)

  return mine



def train(n_epoch, train_losses=[], accs=[], reg_losses=[]):
  """普通のモデル訓練用．（比較のために使う）
  args:
    n_epoch: number of epochs
    train_losses: list of training losses
    accs: list of accuracies
    reg_losses: list of regularizer losses
  """
  # 記録用リストの整理 (list でない場合は list に変換)
  if type(train_losses) != list:
    train_losses = train_losses.tolist()
  if type(accs) != list:
    accs = accs.tolist()
  if type(reg_losses) != list:
    reg_losses = reg_losses.tolist()
  fig = None

  for epoch in range(n_epoch):
    # if epoch in EPOCHS_LRCHANGE:
    #     new_lr = L_RATES[1+EPOCHS_LRCHANGE.index(epoch)]
    #     optimizer.learning_rate = new_lr
    #     print('learning rate was changed to {}.'.format(new_lr))

    start_time = time.time()
    print(f'epoch {epoch+1}', end='')
    # generating signals and splitting
    input_signal_stack, teacher_signal_stack, input_transient, teacher_transient = get_input(
        task, gp)

    n_batch = len(teacher_signal_stack)
    x0_zero = tf.zeros(shape=[gp.bs, gp.N], dtype=gp.mydtype)
    last_state = x0_zero
    if task.transient > 0:
      _, _, last_state = transient(model, task, last_state)

    for i, (inp, tea) in enumerate(zip(input_signal_stack,
                                       teacher_signal_stack,
                                       )):

      z, x, last_state = train_step(
          model, inp, tea, model_metrics, initial_state=last_state)  # ここで訓練１バッチ回す. 前回の状態を引き継ぐ
      if i % 1 == 0:
        print('.', end='')

    elapsed_time = time.time() - start_time  # 時間計測
    n_batch = gp.time_learn // gp.split_length
    batch_per_sec = n_batch/elapsed_time  # 1秒あたりのバッチ数

    # 途中経過の出力
    print(f'\rEpoch {epoch+1}')
    print(f'Regression Loss: {model_metrics.train_loss.result():.5f}')
    if model.use_regularizer:
      print(
          f'weight regularizer Loss: {model_metrics.reg_loss.result():.5f}')
    print(f'Total Loss: {model_metrics.total_loss.result():.5f}')
    print(f'Accuracy: {model_metrics.train_acc.result()}')
    print(f'grad norm: {model_metrics.total_grad.result()}')
    print('Throughput: {:.2f}[sec/epoch] {:.4f}[batch/sec]'.format(
        elapsed_time, batch_per_sec))

    #log
    accs.append(model_metrics.train_acc.result())
    train_losses.append(model_metrics.train_loss.result())
    reg_losses.append(model_metrics.reg_loss.result())
    # 次のエポック用にメトリクスをリセット
    model_metrics.reset_all_states()
    # save
    if (epoch + 1) % 2 == 0:
      save_path = os.path.join(gp.checkpoint_dir, 'ckpt-current')
      model.save_weights(save_path)
    # state をリセット　
    # model.reset_states()
  save_path = os.path.join(gp.checkpoint_dir, 'ckpt-final')
  model.save_weights(save_path)
  print('Training Done')
  return [np.array(train_losses), np.array(accs), np.array(reg_losses)]


def train_with_mine(n_epoch, n_mine=10, losses=None, 
                    model=None, 
                    mine=None, 
                    task=None,
                    model_optimizer=None,
                    mine_optimizer=None,
                    w_recorder=None,
                    gp=None, 
                    model_metrics=None, 
                    mi_metric=None,
                    debug_mode=True):
  # 記録用リストの整理(list でない場合は list に変換)
  if losses is None:
    losses = {'total': [], 'task': [], 'accs': [],
              'regularizer': [], 'mi_before': [], 'mi_after': []}
  if type(losses['total']) != list:
    losses['total'] = losses['total'].tolist()
  if type(losses['task']) != list:
    losses['task'] = losses['task'].tolist()
  if type(losses['accs']) != list:
    losses['accs'] = losses['accs'].tolist()
  if type(losses['regularizer']) != list:
    losses['regularizer'] = losses['regularizer'].tolist()
  if type(losses['mi_before']) != list:
    losses['mi_before'] = losses['mi_before'].tolist()
  if type(losses['mi_after']) != list:
    losses['mi_after'] = losses['mi_after'].tolist()
  fig = None
  fig2 = None
  fig3 = None
  fig4 = None

  # wrapper function for training
  mine_train_step_1 = tf.function(mine_train_step)
  model_train_step_with_mine_1 = tf.function(model_train_step_with_mine)

  #
  w_recorder.record(model, 0)  # 重みの記録用

  for epoch in range(n_epoch):
    # if epoch in EPOCHS_LRCHANGE:
    #     new_lr = L_RATES[1+EPOCHS_LRCHANGE.index(epoch)]
    #     optimizer.learning_rate = new_lr
    #     print('learning rate was changed to {}.'.format(new_lr))
    start_time = time.time()
    print(f'\rEpoch {epoch+1}')

    # MINE net training (n_mine times)
    for nm in range(n_mine):
      #train mine
      mi_metric.reset_states()
      x0_zero = tf.zeros(shape=[gp.bs, gp.N],
                         dtype=gp.mydtype)  # initial state

      mine_train_epoch(model, mine, task, initial_state=x0_zero,
                       mine_optimizer=mine_optimizer,
                       gp=gp, metric_mi=mi_metric, 
                       mine_tsf=mine_train_step_1,
                       )
      # model_train_epoch_with_mine(model, mine, task,
      #       model_optimizer, gp,
      #       model_metrics=model_metrics, metric_mi=mi_metric,
      #       use_mi=True,
      #       )
    print(f'mine net training: done')
    print(f'estimated MI: {mi_metric.result().numpy()}')
    losses['mi_before'].append(mi_metric.result().numpy())
    mi_metric.reset_states()

    # training RNN with mine
    model_train_epoch_with_mine(model, mine, task,
                                model_optimizer, gp,
                                model_metrics=model_metrics, metric_mi=mi_metric,
                                use_mi=True,
                                model_tsf=model_train_step_with_mine_1,
                                )
    elapsed_time = time.time() - start_time  # 時間計測
    n_batch = gp.time_learn // gp.split_length
    batch_per_sec = n_batch/elapsed_time  # 1秒あたりのバッチ数

    # 途中経過の出力
    print('')
    print(f'Regression Loss: {model_metrics.train_loss.result():.5f}')
    if model.use_regularizer:
      print(
          f'Weight regularizer Loss: {model_metrics.reg_loss.result():.5f}')
    print(f'MINE Loss: {model_metrics.mine_loss.result():.5f}')
    print(f'Total Loss: {model_metrics.total_loss.result():.5f}')
    if gp.task_name == 'wm':
      print(f'Accuracy: {model_metrics.train_acc.result()}')

    print(f'grad norm: {model_metrics.total_grad.result()}')
    print('Throughput: {:.2f}[sec/epoch] {:.4f}[batch/sec]'.format(
        elapsed_time, batch_per_sec))

    #log
    if gp.task_name == 'wm':
      losses['accs'].append(model_metrics.train_acc.result().numpy())

    losses['total'].append(model_metrics.total_loss.result().numpy())
    losses['task'].append(model_metrics.train_loss.result().numpy())
    losses['regularizer'].append(model_metrics.reg_loss.result().numpy())
    losses['mi_after'].append(model_metrics.mine_loss.result().numpy())
    # 次のエポック用にメトリクスをリセット
    model_metrics.reset_all_states()
    # save
    if (epoch + 1) % 5 == 0:
      save_path = os.path.join(gp.checkpoint_dir, 'ckpt-current')
      model.save_weights(save_path)

    if ((epoch + 1) % 25 == 0):
      # weight recorder
      w_recorder.record(model, epoch+1)

    # output figures for monitoring
    if debug_mode and ((epoch + 1) % 25 == 0):
      if fig is not None:
        fig.clf()
        plt.close(fig)

      fig = fig_w_rec(model)
      plt.show()

      if fig3 is not None:
        fig3.clf()
        plt.close(fig3)
      fig3 = plt.figure()
      ax = fig3.add_subplot(111)
      ax.plot(np.abs(model.W_in.numpy().T))
      if gp.use_global_feedback:
        ax.plot(np.abs(model.W_fb.numpy().T))
      plt.show()
      if fig4 is not None:
        fig4.clf()
        plt.close(fig4)
      fig4 = plt.figure()
      ax = fig4.add_subplot(111)
      ax.plot(np.abs(model.W_out.numpy()))
      plt.show()

      if fig2 is not None:
        fig2.clf()
        plt.close(fig2)

      fig2 = plt.figure()
      ax = fig2.add_subplot(111)
      ax.plot(losses['mi_before'], label='mi')
      ax.plot(losses['total'], label='total')
      ax.plot(losses['regularizer'], label='l2-reg')
      ax.set_ylim([0, np.max(losses['mi_before'])])
      ax.legend()
      plt.show()

  save_path = os.path.join(gp.checkpoint_dir, 'ckpt-final')
  model.save_weights(save_path)
  print('Training Done')

  return losses


def fig_w_rec(model, ):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.imshow(np.abs(model.W_rec.numpy().T), cmap='Blues')
  return fig



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
    from IPython import get_ipython
    env_name = get_ipython().__class__.__name__
    if env_name == 'TerminalInteractiveShell':
        # IPython shell
        return False
    # Jupyter Notebook
    return True

  if not is_env_notebook():
    print('use AGG')
    mpl.use('Agg')  # off-screen use of matplotlib

  # %% back-propagation learning

  # task_name = TASK_NAME
  set_randseed(SEED)  # for mfs example

  if TASK_NAME == 'wm':
    import gp_wm as gp
  elif TASK_NAME == 'mfs':
    import gp_mfs as gp
  elif TASK_NAME == 'rossler':
    import gp_rossler as gp
  else:
    print('Error: task_name is not correct')
    exit(1)

  #%% prepare input signal
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

  elif gp.task_name == 'periodic':
    task = tasks.PeriodicSignalTask(batch_size=gp.bs, transient=0,
                                    time_learn=3600, time_test=3600,
                                    period=1200.0, random_phase=False,
                                    noise_fb=4e-2)
  #split_length = 2000


  #%% show input and target signals for working memory task
  if DEBUG_MODE:
    if gp.task_name == 'wm':
      plot_wm_signal(task)
    elif gp.task_name == 'mfs':
      plt.plot(task.t_transient, task.signals_transient()[0][0, :, :])
    elif gp.task_name == 'rossler':
      plt.plot(task.t_transient, task.signals_transient()[0][0, :, :])


  #%% model building

  # create model (task-dependent)

  model = set_model_for_task(gp, task)

  #%% setup MINE network

  mine = setup_mine(model, task, gp)

  #%% 動かすチェック
  # mine_train_epoch(model, mine, task=task,
  #                  initial_state=x0, mine_optimizer=mine_optimizer, metric_mi=mi_metric, gp=gp)
  # print(mi_metric.result())


  # %% setting optimizer
  # model_optimizer = tf.keras.optimizers.Adam(
  #     learning_rate=gp.learning_rate, clipnorm=4.0)
  model_optimizer = tf.keras.optimizers.RMSprop(
      learning_rate=gp.learning_rate, clipnorm=gp.clip_grad)
  # mine_optimizer = tf.keras.optimizers.Adam(learning_rate=gp.learning_rate_mine, clipnorm=4.0)
  mine_optimizer = tf.keras.optimizers.RMSprop(
      learning_rate=gp.learning_rate_mine, clipnorm=gp.clip_grad_mine)
  # optimizer = tf.keras.optimizers.SGD(
  #     learning_rate=0.01, clipnorm=10.0)
  # optimizer = tf.keras.optimizers.SGD(
  #     learning_rate=0.01, clipnorm=10.0)

  # setting metrics
  if gp.task_name == 'wm':
    model_metrics = WMMetrics()
  elif gp.task_name == 'mfs':
    model_metrics = MFSMetrics()
  elif gp.task_name == 'rossler':
    model_metrics = RosslerMetrics()
  mi_metric = tf.keras.metrics.Mean(name='mi_metric')
  mi_metric_test = tf.keras.metrics.Mean(name='mi_metric_test')
  w_recorder = WeightRecorder()

  #%% check abs of eigenvalues of W
  if DEBUG_MODE:
    # 行列の固有値分布をみている
    W = model.W_rec.numpy()
    lambdas = np.linalg.eigvals(W)
    plt.plot(np.abs(lambdas))
    plt.xlabel('i')
    plt.ylabel('abs. of i-th eigenvalue of W_rec')
    print(np.abs(lambdas[0]))

  #%% 学習が動くか(train_stepによって重みが変わっているかのチェック)
  if DEBUG_MODE:

    input_signal_stack, teacher_signal_stack, inputs_transient, target_transient = get_input(
        task, gp)  # set input and target signals
    # random initial values
    x0 = tf.random.normal(shape=[gp.bs, gp.N], dtype=gp.mydtype)

    z_tr, x_tr, last_state = transient(
        model, inputs_transient, x0)  # transient run

    W_rec_before = model.W_rec.numpy()
    W_out_before = model.W_out.numpy()
    W_in_before = model.W_in.numpy()
    W_fb_before = model.W_fb.numpy()

    print(model.W_rec[0, 0:100])
    print(model.W_out[0, :])
    #x_tr, z_tr = train_step_external(model, input_signal_stack[0], teacher_signal_stack[0], fb_signal_stack[0], model_metrics)
    x_tr, z_tr, last_state = train_step(
        model, inputs=input_signal_stack[0], teacher=teacher_signal_stack[0],
        model_metrics=model_metrics, optimizer=model_optimizer, initial_state=last_state)  # normal learning
    print(model.W_rec[0, 0:100])
    print(model.W_out[0, :])
    W_rec_after = model.W_rec.numpy()
    W_out_after = model.W_out.numpy()
    W_in_after = model.W_in.numpy()
    W_fb_after = model.W_fb.numpy()

    print(f'modification of W_rec: {(W_rec_before != W_rec_after).all() }')
    print(f'modification of W_in: {(W_in_before != W_in_after).all() }')
    print(f'modification of W_out: {(W_out_before != W_out_after).all() }')
    print(f'modification of W_fb: {(W_fb_before != W_fb_after).all() }')
  #%%
  #   model_train_step_with_mine(model, mine, input_signal_stack[0], teacher_signal_stack[0],
  #                              initial_state=last_state, model_optimizer=model_optimizer, gp=gp,
  #                              model_metrics=model_metrics, metric_mi=mi_metric,
  #                              use_mi=True)
  # #%% check(後で消す)
  #   model_train_epoch_with_mine(model, mine, task,
  #                               model_optimizer,
  #                               gp,
  #                               model_metrics=model_metrics, metric_mi=mi_metric,
  #                               use_mi=True,
  #                               )

  #%% initial MINE learning phase
  print('Starting initial training of MINE statistic net')
  print('')

  mine_tsf = tf.function(mine_train_step)
  mine_initial_train(model, mine, task, mine_optimizer, gp, mi_metric, mine_tsf=mine_tsf)

  #%% save initial model
  save_path = os.path.join(gp.checkpoint_dir, 'ckpt-initial')
  model.save_weights(save_path)
  #TODO: MINEの状態も保存する

  # %%
  # define training loop


  #%% set of list for recording losses
  losses = {'total': [], 'task': [], 'accs': [],
            'regularizer': [], 'mi_before': [], 'mi_after': []}
  #%% run training

  losses = train_with_mine(gp.epochs, n_mine=gp.n_mine_pre, losses=losses, 
                    model=model, 
                    mine=mine, 
                    task=task,
                    gp=gp,     
                    model_optimizer=model_optimizer,
                    mine_optimizer=mine_optimizer,
                    w_recorder=w_recorder,
                    model_metrics=model_metrics, 
                    mi_metric=mi_metric)
  

  print('-----------------')
  print('training done')
  print('-----------------')
  #%% save losses
  if gp.task_name == 'wm':
    data_dir = './data/wm'
  elif gp.task_name == 'mfs':
    data_dir = './data/mfs'
  elif gp.task_name == 'rossler':
    data_dir = './data/rossler'

  os.makedirs(data_dir, exist_ok=True)

  with open(os.path.join(data_dir, 'losses.pkl'), 'wb') as f:
    pickle.dump(losses, f)
  #%% save weights
  with open(os.path.join(data_dir, 'w_recorder.pkl'), 'wb') as f:
    pickle.dump(w_recorder, f)


  # %%
  if gp.task_name == 'wm':
    fig_dir = './figures/wmfigs'
  elif gp.task_name == 'mfs':
    fig_dir = './figures/mfsfigs'
  elif gp.task_name == 'rossler':
    fig_dir = './figures/rosslerfigs'

  os.makedirs(fig_dir, exist_ok=True)

  fig = plt.figure()
  ax1 = fig.add_subplot(111)
  ln1 = ax1.plot(losses['total'], 'C0', label='total loss')
  h1, l1 = ax1.get_legend_handles_labels()

  #ax1.legend()

  if gp.task_name == 'wm':
    ax2 = ax1.twinx()
    ln2 = ax2.plot(losses['accs'], 'C1', label='accuracy')
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='center right')
    ax2.set_ylabel('accuracy')
  else:
    ax1.legend(h1, l1, loc='center right')

  ax1.set_xlabel('epoch')
  ax1.set_ylabel('loss')

  fig.savefig(os.path.join(fig_dir, 'train_loss.png'), dpi=200)

  plt.show()
  #%%
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ln1 = ax.plot(losses['task'], 'C0', label='task loss')
  ax.legend()
  ax.set_xlabel('epoch')
  ax.set_ylabel('loss')
  fig.savefig(os.path.join(fig_dir, 'task_loss.png'), dpi=200)

  plt.show()
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ln1 = ax.plot(losses['regularizer'], 'C0', label='weight regularizer loss')
  ax.legend()
  ax.set_xlabel('epoch')
  ax.set_ylabel('loss')
  fig.savefig(os.path.join(fig_dir, 'regularization_loss.png'), dpi=200)

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ln1 = ax.plot(losses['mi_before'], 'C0', label='MI estimation')
  # ln2 = ax.plot(losses['mi_after'], 'C1', label='MI after train step')
  ax.set_ylim([0, 5])

  ax.legend()
  ax.set_xlabel('epoch')
  ax.set_ylabel('loss')
  fig.savefig(os.path.join(fig_dir, 'mi_loss.png'), dpi=200)


  #%% load model
  save_path = os.path.join(gp.checkpoint_dir, 'ckpt-final')
  model.load_weights(save_path)


  #%% transient-training

  t_te_start = time.time()

  task.gen_signals()
  inputs_transient, target_transient = task.signals_transient()
  inputs, target = task.signals_learning()
  input_stack = split_sequence(inputs, gp.split_length)
  input_stack, target_stack, inputs_transient, target_transient = get_input(
      task, gp)
  x0 = tf.random.normal(shape=[gp.bs, gp.N], dtype=gp.mydtype)

  z_tr, x_tr, last_state = transient(model, inputs_transient, x0)
  z, r1, r2 = run_and_extract_group(model, input_stack[0], last_state, gp)
  elapsed3 = time.time() - t_te_start
  #%%

  #x_training, z_training = model.gen_sequence_w_external_fb(input_signal, fb_signal + 0.1*tf.random.uniform(shape = fb_signal.shape))
  #z_training = model(input_signal, initial_state=last_state)
  #x_training = model.x

  if (gp.task_name == 'mfs') or (gp.task_name == 'rossler'):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(task.t_learn[0:gp.split_length],
            z[0, :, :], '-', label='prediction')
    ax.plot(task.t_learn[0:gp.split_length],
            input_stack[0][0, :, :], '-', label='input')
    fig.savefig(os.path.join(fig_dir, 'trajectory.png'), dpi=200)
  #%%

  fig = plt.figure()
  ax = fig.add_subplot(111)
  label_win = [f'W_in_{n}' for n in range(model.W_in.shape[0])]
  ax.plot(model.W_in.numpy().T, label=label_win)
  ax.legend()
  fig.savefig(os.path.join(fig_dir, 'w_in.png'), dpi=200)
  plt.show()
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(model.W_out.numpy())
  ax.legend()
  fig.savefig(os.path.join(fig_dir, 'w_out.png'), dpi=200)
  plt.show()

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(model.W_fb.numpy().T)
  ax.legend()
  fig.savefig(os.path.join(fig_dir, 'w_fb.png'), dpi=200)
  plt.show()


  #%%
  fig = fig_w_rec(model)
  fig.savefig(os.path.join(fig_dir, 'w_rec.png'), dpi=200)
  plt.show()

  #%% for wm task
  if gp.task_name == 'wm':
    dpi = 300
    os.makedirs(fig_dir, exist_ok=True)

    def fig_wm_task(task, z_trans, z_training, n_mem=2, ns=0, figsize=(7, 6)):
      fig, axes = plt.subplots(2*n_mem, 1, figsize=figsize)
      tmax = 3000
      for mem in range(n_mem):
        ax = axes[2*mem]
        ax.plot(task.t_total[0:tmax], 1+0.75 *
                task.input_signal[ns, 0:tmax, 2*mem+0], 'r-', label='on')
        ax.plot(task.t_total[0:tmax], 0.75*task.input_signal[ns,
                0:tmax, 2*mem+1], 'k-', label='off')
        ax.tick_params(labelleft=False, labelbottom=False)
        ax = axes[2*mem+1]
        if task.transient > 0:
          ax.plot(task.t_transient,
                  z_trans[ns, :, mem], '-', label='transient')
        ax.plot(task.t_total[task.transient:tmax], z_training[ns, 0:(tmax-task.transient),
                                                              mem], '-', label='test')
        ax.plot(task.t_total[0:tmax],
                task.target_signal[ns, 0:tmax, mem], '-', label='target')
        ax.legend(bbox_to_anchor=(1.0, 0.), loc='lower left')

      return fig

    fig = fig_wm_task(task, z_tr.numpy(), z.numpy(),
                      n_mem=2, ns=0, figsize=(8, 4))
    #fig.savefig(os.path.join(fig_dir, 'wm_traj.png'), dpi=dpi)
    fig.savefig(os.path.join(fig_dir, 'trajectory.png'), dpi=200)
    plt.show()


