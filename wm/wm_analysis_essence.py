# 結果の分析． 本質的な部分だけを使う．（レガシー的な部分をを取り除く）
# 他のファイルから使うことを意識する．（関数に分解）
#%%

# 注意点（行列の向き）
# 再帰結合行列W は内部的には xは横ベクトルとしてx*W の形で使っている，
# つまり i,j要素は ニューロンi-> j の結合を意味する．
# 図にするときはこれを転置している．つまり(i,j)要素は j->iの結合を表す．

# 横一行は　ひとつのニューロンへの入力
# 縦一列は　ひとつのニューロンへの出力


from mine import run_and_extract_group
from tools import split_sequence
from parameters import  WeightRecorder
import tasks
from rnnmodel import LeakyRNNModel, build_rnn_model, transient
import scipy.stats
from sklearn.decomposition import PCA
from numpy import linalg as LA  # linear algebra
import time
import os
from tensorflow.keras import Model
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pickle


#TASK_NAME = 'wm'
debug_mode = True
# import gp_wm as gp
targ_colors= ['dodgerblue', 'orchid', 'orangered']


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

# mydtype = tf.float32

#tf.keras.backend.set_floatx('float32')
#mydtype = tf.float32


def _run_once(task, model, gp):
  task.gen_signals()
  inputs_transient, _ = task.signals_transient()
  input, target = task.signals_learning()
  input_stack = split_sequence(input, gp.split_length)
  target_signal_stack = split_sequence(target, gp.split_length)

  x0 = tf.random.normal(shape=[gp.bs, gp.N], dtype=gp.mydtype)
  z_tr, x_tr, last_state = transient(model, inputs_transient, x0)
  z, r1, r2 = run_and_extract_group(model, input_stack[0], last_state, gp)
  return [input, target], [input_stack, target_signal_stack], [z_tr, x_tr], [z, r1, r2]


def plot_wm_signal(task, length=5000):
  input_signal, teacher_signal, fb_signal = task.signals_learning(fb=True)
  n_ax = 3 * task.n_mem

  fig, axes = plt.subplots(n_ax, 1, figsize=(7, 4))
  for m in range(task.n_mem):
    axes[3*m].plot(input_signal[0, 0:length, 2*m], '-k')
    axes[3*m].tick_params(labelbottom=False, labelleft=False)
    axes[3*m+1].plot(input_signal[0, 0:length, 2*m+1], '-r')
    axes[3*m+1].tick_params(labelbottom=False, labelleft=False)
    axes[3*m+2].plot(teacher_signal[0, 0:length, m])
    axes[3*m+2].tick_params(labelbottom=False)
  axes[3*task.n_mem-1].tick_params(labelbottom=True)
  return fig


def fig_wm_task(task, z_trans, z_training, n_mem=2, ns=0, tmax=3000, figsize=(7, 6)):
  fig, axes = plt.subplots(2*n_mem, 1, figsize=figsize)

  for mem in range(n_mem):
    ax = axes[2*mem]
    ax.plot(task.t_total[0:tmax], 1+0.75 *
            task.input_signal[ns, 0:tmax, 2*mem+0], 'r-', label='on')
    ax.plot(task.t_total[0:tmax], 0.75*task.input_signal[ns,
            0:tmax, 2*mem+1], 'k-', label='off')
    ax.tick_params(labelleft=False, labelbottom=False)
    ax = axes[2*mem+1]
    if task.transient > 0:
      ax.plot(task.t_transient, z_trans[ns, :, mem], '-', label='transient')
    ax.plot(task.t_total[task.transient:tmax], z_training[ns, 0:(tmax-task.transient),
                                                          mem], '-', label='test')
    ax.plot(task.t_total[0:tmax],
            task.target_signal[ns, 0:tmax, mem], '-', label='target')
    ax.legend(bbox_to_anchor=(1.0, 0.), loc='lower left')

  return fig


def plot_mfs_signal(task, length=5000):
  input_signal, teacher_signal = task.signals_learning(fb=False)
  n_ax = task.N_in

  fig, axes = plt.subplots(n_ax, 1, figsize=(7, 4))
  for m in range(n_ax):
    axes[m].plot(input_signal[0, 0:length, m], '-k', label='input')
    axes[m].plot(teacher_signal[0, 0:length, m], '-r', label='target')
    axes[m].tick_params(labelbottom=True, labelleft=False)
    axes[m].legend()
  return fig


def fig_mfs_task(task, z_tr, z_training, ns=0, figsize=(8, 4)):
  n_in = task.N_in
  fig, axes = plt.subplots(n_in, 1, figsize=figsize)
  tmax = 2000

  for m in range(n_in):
    ax = axes[m]
    ax.plot(task.t_total[0:tmax],
            task.target_signal[ns, 0:tmax, m], 'k-', label='target')

    if task.transient > 0:
      ax.plot(task.t_transient, z_tr[ns, :, m], '-', label='transient')
    ax.plot(task.t_total[task.transient:tmax], z_training[ns, 0:(tmax-task.transient),
                                                          m], '-', label='training')

    ax.tick_params(labelleft=False, labelbottom=False)

    # ax.plot(task.t_total[0:tmax],
    #         task.target_signal[ns, 0:tmax, mem], '-', label='target')
    ax.legend(bbox_to_anchor=(1.0, 0.), loc='lower left')

  return fig


def make_fig_losses(losses, gp, fig_dir):

  fig = plt.figure(figsize=(8, 6))
  ax1 = fig.add_subplot(111)
  ln1 = ax1.plot(losses['total'], 'C0', label='total loss')
  h1, l1 = ax1.get_legend_handles_labels()
  if gp.task_name == 'wm':
    ax2 = ax1.twinx()
    ln2 = ax2.plot(losses['accs'], 'C3', label='accuracy')
    ax2.set_ylim([0.5, 1.0])
    ax2.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='center right')
    ax2.set_ylabel('accuracy')
  else:
    ax1.legend(h1, l1, loc='center right')

  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Total Loss')

  fig.savefig(os.path.join(fig_dir, 'train_loss.png'),
              bbox_inches='tight')
  fig.savefig(os.path.join(fig_dir, 'train_loss.pdf'),
              bbox_inches='tight')
  plt.show()
  fig.clf()
  plt.close(fig)

  #
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)
  ln1 = ax.plot(losses['task'], 'C0', label='task loss')
  ln2 = ax.plot(losses['mi_before'], 'C1', label='MI estimation')
  ln3 = ax.plot(losses['regularizer'], 'C2', label='weight regularizer loss')
  ax.legend()
  ax.set_xlabel('Epoch')
  ax.set_ylabel('Loss')
  fig.savefig(os.path.join(fig_dir, 'three_loss.png'), bbox_inches='tight')

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ln1 = ax.plot(losses['regularizer'], 'C0', label='weight regularizer loss')
  ax.legend()
  ax.set_xlabel('epoch')
  ax.set_ylabel('regularization loss')
  fig.savefig(os.path.join(fig_dir, 'regularization_loss.png'),
              bbox_inches='tight')
  fig.savefig(os.path.join(fig_dir, 'regularization_loss.pdf'),
              bbox_inches='tight')

  fig = plt.figure()
  ax = fig.add_subplot(111)

  ln2 = ax.plot(losses['mi_before'], 'C1', label='MI estimation')

  # ln2 = ax.plot(losses['mi_after'], 'C1', label='MI after train step')

  ax.legend()
  ax.set_xlabel('epoch')
  ax.set_ylabel('MI')
  fig.savefig(os.path.join(fig_dir, 'mi_loss.png'), bbox_inches='tight')
  fig.savefig(os.path.join(fig_dir, 'mi_loss.pdf'), bbox_inches='tight')  
  plt.show()


def draw_pca_trajectory(ax, Y):
  sn = 0
  ax.plot(Y[sn, :, 0], Y[sn, :, 1])
  ax.set_xlabel('PC 1')
  ax.set_ylabel('PC 2')

# histogram


def draw_pca_hist2d(ax, Y):
  ax.hist2d(Y[:, :, 0].flatten(), Y[:, :, 1].flatten(), bins=[100, 100])
  ax.set_xlabel('PC 1')
  ax.set_ylabel('PC 2')


def draw_pca_3d(ax, Y):
  sn = 0
  ax.plot(Y[sn, :, 0], Y[sn, :, 1], Y[sn, :, 2])
  ax.set_xlabel('PC 1')
  ax.set_ylabel('PC 2')
  ax.set_zlabel('PC 3')


def W_part(W, gp):
  w11 = W[gp.g1lim[0]:gp.g1lim[1], gp.g1lim[0]:gp.g1lim[1]]
  w12 = W[gp.g1lim[0]:gp.g1lim[1], gp.g2lim[0]:gp.g2lim[1]]
  w21 = W[gp.g2lim[0]:gp.g2lim[1], gp.g1lim[0]:gp.g1lim[1]]
  w22 = W[gp.g2lim[0]:gp.g2lim[1], gp.g2lim[0]:gp.g2lim[1]]
  return [w11, w12, w21, w22]


def apply_pca(X, n_comp=40):
  X_shape = X.shape
  Xr = X.reshape((X_shape[0]*X_shape[1], X_shape[2]))  # (bs*t_length, N ))

  pca = PCA(n_components=n_comp)
  try: 
    Yr = pca.fit_transform(Xr)  # (bs*t_length, 3)
    Y = Yr.reshape((X_shape[0], X_shape[1], n_comp))  # (bs, t_length, 3)
  except ValueError as e:
    print(f"Error in PCA: {e}")
    # If PCA fails, return the original data
    Y = X
    pca = None
  return Y, pca


def corr_x_and_target(X, target):
  ''' calculate correlations between X and X, and X and target
  args
  X: (bs, t_length, N)
  target: (bs, t_length, N_out)
  '''
  n_out = target.shape[2]
  Xr = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))  # (bs*t_length, N ))
  target_r = target.reshape((X.shape[0]*X.shape[1], target.shape[2]))
  cor_mat = np.corrcoef(target_r.T, Xr.T)  # correlation matrix (42x42)
  corr_X = cor_mat[n_out:, n_out:]  # (N,N)
  # correlation to target
  corr_targ_X = cor_mat[0:n_out, n_out:]  # (N_out, N)
  return corr_X, corr_targ_X



def draw_dif_corr(ax, dif_corr):
  ax.set_xlabel('neuron index')
  ax.set_ylabel('$d_{i} = |c^{1}_{i}|-|c^{2}_{i}|}$')
  ax.plot(dif_corr)


def corr_sorting(corr_X, corr_targ_X):

  # difference of target_correlation ( 0 or 1)
  dif_corr = np.abs(corr_targ_X[0, :]) - np.abs(corr_targ_X[1, :])
  # sorting index (descending order)
  cor_sort_index = np.argsort(dif_corr)[::-1]

  # sorted difference of correlation
  dif_corr_sorted = dif_corr[cor_sort_index]

  # sort based on diff of corr strength
  corr_xx_sorted = corr_X[:, cor_sort_index]
  corr_xx_sorted = corr_xx_sorted[cor_sort_index, :]
  corr_targ_sorted = corr_targ_X[:, cor_sort_index]

  return cor_sort_index, dif_corr_sorted, corr_xx_sorted, corr_targ_sorted


def weights_to_numpy(model):
  W_rec = model.W_rec.numpy()
  W_in = model.W_in.numpy()
  W_out = model.W_out.numpy()
  W_fb = model.W_fb.numpy()
  return (W_in, W_rec, W_out, W_fb)


def get_W_total(model, gp):
  if gp.use_global_feedback:
    W_tot = model.W_rec.numpy() + model.W_out.numpy()@model.W_fb.numpy()
  else:
    W_tot = model.W_rec.numpy()
  return W_tot


def load_model_and_task(gp):
  model = build_rnn_model(gp)

  if task_name == 'wm':
    task = tasks.WorkingMemoryTask(batch_size=gp.bs,
                                   n_mem=gp.n_mem, len_pulse=40, freq_pulse=1.0/500.0, slope=0.1,
                                   transient=gp.transient, time_learn=gp.time_learn, time_test=0, )
  elif task_name == 'mfs':
    task = tasks.MultiFrequencySinusoidalTask(freqs=gp.freqs,
                                              random_phase=True, use_input=True, pred_step=1,
                                              batch_size=gp.bs, transient=gp.transient,
                                              time_learn=gp.time_learn, time_test=0,
                                              )
  elif task_name == 'periodic':
    task = tasks.PeriodicSignalTask(batch_size=gp.bs, transient=0,
                                    time_learn=3600, time_test=3600,
                                    period=1200.0, random_phase=False,
                                    noise_fb=4e-2)
  else:
    print('Task name is not valid')
    return
  #split_length = 2000

  # load model
  save_path = os.path.join(gp.checkpoint_dir, 'ckpt-final')
  model.load_weights(save_path)

  # run once
  t_te_start = time.time()

  [input, target], [input_stack, target_signal_stack], [
      z_tr, x_tr], [z, r1, r2] = _run_once(task, model, gp)

  # z = model(input_stack[0], x0)  # test run
  model.additional_setup()
  model.summary()
  return model, task


def set_fig_style(gp):
  plt.rcParams["font.size"] = 20
  plt.rcParams['savefig.dpi'] = 300
  if gp.task_name == 'wm':
    fig_dir = './figures/wmfigs'
  elif gp.task_name == 'mfs':
    fig_dir = './figures/mfsfigs'
  else:
    fig_dir = './figures'

  dpi = 300
  os.makedirs(fig_dir, exist_ok=True)
  return fig_dir, dpi


def make_fig_task(model, task, gp, fig_dir):
  ''' make figure explains task and input output of model
    args:
      model: model
      task: task
      gp: global parameters
      fig_dir: figure directory
  '''
  [input, target], [input_stack, target_signal_stack], [
      z_tr, x_tr], [z, r1, r2] = _run_once(task, model, gp)
  if gp.task_name == 'wm':
    fig = plot_wm_signal(task, length=5000)
    fig.savefig(os.path.join(fig_dir, 'wm_task.png'),
                bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, 'wm_task.pdf'),
                bbox_inches="tight")
    plt.show()
    fig.clf()
    plt.close(fig)

    fig = fig_wm_task(task, z_tr.numpy(), z.numpy(),
                      n_mem=2, ns=0, figsize=(10, 8))
    fig.savefig(os.path.join(fig_dir, 'wm_traj.png'),
                bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, 'wm_traj.pdf'),
                bbox_inches="tight")

    plt.show()
    fig.clf()
    plt.close(fig)
  elif gp.task_name == 'mfs':
    fig = plot_mfs_signal(task, length=500)
    fig.savefig(os.path.join(fig_dir, 'mfs_signal.png'),
                bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, 'mfs_signal.pdf'),
                bbox_inches="tight")
    plt.show()
    fig.clf()
    plt.close(fig)
    fig = fig_mfs_task(task, z_tr.numpy(), z.numpy())
    fig.savefig(os.path.join(fig_dir, 'mfs_task.png'))
    fig.savefig(os.path.join(fig_dir, 'mfs_task.pdf'))
    plt.show()
    fig.clf()
    plt.close(fig)
  return


def make_fig_PCA(model, task, gp, fig_dir):
  # to numpy
  X = model.x.numpy()
  R = model.r.numpy()
  Z = model.z.numpy()
  bs = X.shape[0]
  t_length = X.shape[1]
  #N = X.shape[2]

  # PCA

  # if NaN in X, replace with 0
  if np.isnan(X).any():
    print("NaN found in X, replacing with 0")
    X = np.nan_to_num(X)

  Y, pca = apply_pca(X, n_comp=2)
  pca_component = pca.components_.T  # (N,3)
  pca_explained_var_ratio = pca.explained_variance_ratio_

  #  PCA figure

  fig = plt.figure(figsize=(5, 5))
  ax = fig.add_subplot(1, 1, 1)
  draw_pca_trajectory(ax, Y)
  fig.savefig(os.path.join(fig_dir, 'pca_traj.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'pca_traj.pdf'),
              bbox_inches="tight")

  fig = plt.figure()
  ax = fig.add_subplot(111)
  draw_pca_hist2d(ax, Y)
  fig.savefig(os.path.join(fig_dir, 'pca_hist2d.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'pca_hist2d.pdf'),
              bbox_inches="tight")
  plt.show()
  fig.clf()
  plt.close(fig)
  return pca_component, pca_explained_var_ratio


def calc_corr_X_target(model, task, gp):
  [input, target], [input_stack, target_signal_stack], [
      z_tr, x_tr], [z, r1, r2] = _run_once(task, model, gp)
  X = model.x.numpy()
  corr_X, corr_targ_X = corr_x_and_target(X, target_signal_stack[0].numpy())
  phi_corr = np.arctan(
      np.abs(corr_targ_X[1, :]) / np.abs(corr_targ_X[0, :]))
  dif_corr = np.abs(corr_targ_X[0, :]) - np.abs(corr_targ_X[1, :])

  return corr_X, corr_targ_X, phi_corr, dif_corr
  # sortingは使わない
  # cor_sort_index, dif_corr_sorted, corr_xx_sorted, corr_targ_sorted = corr_sorting(
  #     corr_X, corr_targ_X)


def draw_corr_12_g1g2(ax, corr_targ_X, gp):
  ax.plot(corr_targ_X[0, gp.g1lim[0]:gp.g1lim[1]], corr_targ_X[1, gp.g1lim[0]:gp.g1lim[1]],
          '.', color='steelblue', label='group 1')
  ax.plot(corr_targ_X[0, gp.g2lim[0]:gp.g2lim[1]], corr_targ_X[1, gp.g2lim[0]:gp.g2lim[1]],
          '.', color='tomato', label='group 2')
  ax.legend(fontsize=20)
  ax.set_xlabel('$c^1_i$', fontsize=20)
  ax.set_ylabel('$c^2_i$', fontsize=20)


def draw_corr_abs_12_g1g2(ax, corr_targ_X, gp):
  ax.plot(np.abs(corr_targ_X[0, gp.g1lim[0]:gp.g1lim[1]]), np.abs(corr_targ_X[1, gp.g1lim[0]:gp.g1lim[1]]),
          'o', color='blue', label='Group 1', markersize=8)
  ax.plot(np.abs(corr_targ_X[0, gp.g2lim[0]:gp.g2lim[1]]), np.abs(corr_targ_X[1, gp.g2lim[0]:gp.g2lim[1]]),
          '^', color='red', label='Group 2', markersize=8)
  # tick fontsize
  ax.tick_params(axis='both', which='major', labelsize=18)
  ax.set_xlabel('$|c^{(1)}_{i}|$', fontsize=24)
  ax.set_ylabel('$|c^{(2)}_{i}|$', fontsize=24)
  ax.legend(fontsize=22)


def draw_corr_matrix(ax, fig, corr_X):
  ima = ax.imshow(np.abs(corr_X), cmap='Reds')
  ax.set_xlabel('neuron index')
  ax.set_ylabel('neuron index')
  ax.set_title('$correlation |c_{ij}|$ ')
  cax = fig.colorbar(ima)
  ax_pos = ax.get_position()
  cax_pos0 = cax.ax.get_position()
  cax_pos1 = [cax_pos0.x0, ax_pos.y0, cax_pos0.x1 -
              cax_pos0.x0, ax_pos.y1 - ax_pos.y0]
  cax.ax.set_position(cax_pos1)


def make_fig_corr_X_targ(corr_X, corr_targ_X, phi_corr, dif_corr, gp, fig_dir, fname='diff_corr'):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  draw_dif_corr(ax, dif_corr)
  fig.savefig(os.path.join(fig_dir, fname + '.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, fname + '.pdf'),
              bbox_inches="tight")
  plt.show()
  fig.clf()
  plt.close(fig)

  def draw_corr_target(ax, cor_targ_X):

    ax.set_xlabel('neuron index')
    ax.set_ylabel('$|c^{j}_{i}|$')
    if corr_targ_X.shape[0] ==2:
      pl_label = ['j=1', 'j=2']
    else:
      pl_label = ['j=1', 'j=2', 'j=3']      
    for j in range(corr_targ_X.shape[0]):
      ax.plot(np.abs(corr_targ_X[j,:]), color=targ_colors[j], label=pl_label[j])
    ax.legend()

  fig = plt.figure()
  ax = fig.add_subplot(111)
  draw_corr_target(ax, corr_targ_X)
  fig.savefig(os.path.join(fig_dir, 'corr_targ.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'corr_targ.pdf'),
              bbox_inches="tight")
  plt.show()
  fig.clf()
  plt.close(fig)

  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(111)
  draw_corr_12_g1g2(ax, corr_targ_X, gp)
  fig.savefig(os.path.join(fig_dir, 'corr_12_g1g2.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'corr_12_g1g2.pdf'),
              bbox_inches="tight")
  plt.show()
  fig.clf()
  plt.close(fig)

  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(111)
  draw_corr_abs_12_g1g2(ax, corr_targ_X, gp)
  fig.savefig(os.path.join(fig_dir, 'corr_abs_12_g1g2.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'corr_abs_12_g1g2.pdf'),
              bbox_inches="tight")
  plt.show()
  fig.clf()
  plt.close(fig)
  #

  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111)
  draw_corr_matrix(ax, fig,  corr_X)
  fig.savefig(os.path.join(fig_dir, 'corr_matrix.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'corr_matrix.pdf'),
              bbox_inches="tight")

  # if nan is in phi, replace with 0
  if np.isnan(phi_corr).any():
    print("NaN found in phi_corr, replacing with 0")
    phi_corr = np.nan_to_num(phi_corr)
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.hist(phi_corr, bins=50)
  ax.set_xlabel('$\phi_{corr}$') #type:ignore
  fig.savefig(os.path.join(fig_dir, 'hist_phi_corr.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'hist_phi_corr.pdf'),
              bbox_inches="tight")

  plt.show()
  fig.clf()
  plt.close(fig)


def make_fig_W_rec(W_rec, W_rec_nom, fig_dir, fname_base='w_rec'):
  fig = plt.figure(figsize=(14, 14))
  ax = fig.add_subplot(111)
  ax.imshow(np.abs(W_rec.T), cmap='Blues')
  ax.set_title('$|w_{ij}|$ ')
  ax.set_xlabel('j')
  ax.set_ylabel('i')

  fig.savefig(os.path.join(fig_dir, fname_base + '_abs.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, fname_base + '_abs.pdf'),
              bbox_inches="tight")

  fig = plt.figure(figsize=(14, 14))
  ax = fig.add_subplot(111)
  ax.imshow(np.abs(W_rec_nom.T), cmap='Blues')
  ax.set_title('$|w_{ij}|$ ')
  ax.set_xlabel('j')
  ax.set_ylabel('i')
  fig.savefig(os.path.join(fig_dir, fname_base + '_nom.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, fname_base + '_nom.pdf'),
              bbox_inches="tight")

  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111)
  ax.imshow(W_rec.T, cmap='PiYG')
  ax.set_title('$w_{ij}$ ')
  ax.set_xlabel('j')
  ax.set_ylabel('i')
  fig.savefig(os.path.join(fig_dir, fname_base + '_rec.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, fname_base + '_rec.pdf'),
              bbox_inches="tight")

  return


def make_fig_W_summarize(W_rec, W_out, W_in, fig_dir, fname='weight_summary'):
  fig, axes = plt.subplots(2, 2,
                           gridspec_kw={
                               'width_ratios': [8, 1],
                               'height_ratios': [8, 1]}, figsize=(12, 10))

  axes[0, 0].imshow(np.abs(W_rec.T), cmap='Blues')
  axes[0, 0].set_title('$|w_{ij}|$ ')
  axes[0, 0].set_xlabel('j')
  axes[0, 0].set_ylabel('i')

  axes[0, 1].imshow(np.abs(W_in).T, cmap='Blues', aspect=0.5)
  axes[0, 1].tick_params(labelbottom=False, labelleft=False)
  axes[0, 1].set_title('$|w^{in}_{ik}|$ ', loc='center')

  axes[1, 0].imshow(np.abs(W_out).T, cmap='Blues', )
  axes[1, 0].tick_params(labelbottom=False, labelleft=False)
  axes[1, 0].set_ylabel('$|w^{out}_{kj}|$ ')
  l1, b1, w1, h1 = axes[0, 0].get_position().bounds
  lo, bo, wo, ho = axes[1, 0].get_position().bounds

  axes[1, 1].remove()
  li, bi, wi, hi = axes[0, 1].get_position().bounds

  # axes[1, 2].remove()

  fig.tight_layout()

  fig.savefig(os.path.join(fig_dir, fname+'.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, fname+'.pdf'),
              bbox_inches="tight")


def make_fig_diff_out(dif_corr, d_w_out, corrcoef_dc_d_w_out, W_out, fig_dir):
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)
  ax.plot(dif_corr, np.abs(
      W_out[:, 0]), '.', label='$w^{(out)}_{1i}$')
  ax.plot(dif_corr, np.abs(
      W_out[:, 1]), '.', label='$w^{(out)}_{2i}$')
  ax.set_xlabel('$d_i$')
  ax.set_ylabel('$|w^{(out)}_{ji}|$')
  ax.legend()

  fig.savefig(os.path.join(fig_dir, 'corr_dc_w_out.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'corr_dc_w_out.pdf'),
              bbox_inches="tight")
  plt.show()
  fig.clf()
  plt.close(fig)
  # d_w_out vs dif_corr
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)
  ax.plot(dif_corr, d_w_out, '.')
  ax.set_xlabel('$d_i$')
  ax.set_ylabel('$|w^{(out)}_{1i}| - |w^{(out)}_{2i}|$')
  ax.text(0.2, -0.15, f"r = {corrcoef_dc_d_w_out:.3}")
  fig.savefig(os.path.join(fig_dir, 'corr_dc_d_w_out.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'corr_dc_d_w_out.pdf'),
              bbox_inches="tight")
  plt.show()
  fig.clf()
  plt.close(fig)


def calc_d_w_out(W_out, W_out_nom, dif_corr):
  d_w_out = np.abs(W_out[:, 0]) - np.abs(W_out[:, 1])
  corrcoef_dc_d_w_out, p_val = scipy.stats.pearsonr(
      dif_corr, d_w_out)  # dc との相関をみておく．

  d_w_out_nom = np.abs(
      W_out_nom[:, 0]) - np.abs(W_out_nom[:, 1])
  corrcoef_dc_d_w_out_nom, p_val = scipy.stats.pearsonr(
      dif_corr, d_w_out_nom)  # dc との相関をみておく．
  return d_w_out, corrcoef_dc_d_w_out, d_w_out_nom, corrcoef_dc_d_w_out_nom


def calc_d_w_in(W_in, dif_corr, task_name):
  aW_in = np.abs(W_in)
  # sum(abs w_in for mem1)  - sum( abs w_in for mem2)
  if task_name == 'wm':
    d_w_in = np.sum(aW_in[0:2, :], axis=0) - \
        np.sum(aW_in[2:4, :], axis=0)
  elif task_name == 'mfs':
    d_w_in = aW_in[0, :] - aW_in[1, :]
  else:
    d_w_in = aW_in[0, :] - aW_in[1, :]

  corrcoef_dc_d_w_in, p_val = scipy.stats.pearsonr( 
      dif_corr, d_w_in)  # dc との相関をみておく．#type ignore
  return d_w_in, corrcoef_dc_d_w_in


def make_fig_diff_in(dif_corr, d_w_in, corrcoef_dc_d_w_in, W_in, fig_dir):
  aW_in = np.abs(W_in)
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)
  ax.plot(dif_corr, aW_in.transpose(), '.')
  ax.set_xlabel('$d_i$')
  ax.set_ylabel('$|w^{(in)}_{mi}|$ ')
  fig.savefig(os.path.join(fig_dir, 'corr_dc_win.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'corr_dc_win.pdf'),
              bbox_inches="tight")

  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)
  ax.plot(dif_corr, d_w_in.transpose(), '.')
  ax.set_xlabel('$d_i$')
  ax.set_ylabel('diff. of input weights strength')
  ax.text(0.2, -0.5, f"r = {corrcoef_dc_d_w_in:.3}")
  fig.savefig(os.path.join(fig_dir, 'corr_dc_d_w_in.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'corr_dc_d_w_in.pdf'),
              bbox_inches="tight")


def make_fig_weight_hist(W_rec, gp, fig_dir):
  # 重みの分布をみてみる．

  #  全体の分布
  #hw_bins = np.linspace(0,0.002, 101)
  hw_rec, hw_bins = np.histogram(
      np.abs(W_rec.flatten()), bins=100, density=False)

  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)
  ax.plot(hw_bins[1:], hw_rec)
  ax.set_xlabel('|w|')
  fig.savefig(os.path.join(fig_dir, 'wm_hist_w_rec.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'wm_hist_w_rec.pdf'),
              bbox_inches="tight")
  plt.show()
  fig.clf()
  plt.close(fig)

  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)
  ax.plot(hw_bins[1:], np.cumsum(hw_rec)/(gp.N**2))
  ax.set_xlabel('|w|')
  ax.set_ylabel('cumulative density')
  fig.savefig(os.path.join(fig_dir, 'wm_hist_cum_w_rec.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'wm_hist_cum_w_rec.pdf'),
              bbox_inches="tight")
  # 4つに分ける
  W_rec_part = W_part(W_rec, gp)

  hw_rec_part = []
  hw_cum_rec_part = []
  for ii in range(4):
    hw_part, _ = np.histogram(np.abs(W_rec_part[ii].flatten()), bins=hw_bins)
    hw_cum_part = np.cumsum(hw_part)/((gp.N/2)**2)

    hw_rec_part.append(hw_part)
    hw_cum_rec_part.append(hw_cum_part)

  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)
  ax.plot(hw_bins[1:], hw_rec_part[0], label='g1-g1')
  ax.plot(hw_bins[1:], hw_rec_part[1], label='g1-g2')
  ax.plot(hw_bins[1:], hw_rec_part[2], label='g2-g1')
  ax.plot(hw_bins[1:], hw_rec_part[3], label='g2-g2')
  ax.set_xlabel('|w|')
  ax.legend()
  fig.savefig(os.path.join(fig_dir, 'wm_hist_w_rec_part.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'wm_hist_w_rec_part.pdf'),
              bbox_inches="tight")

  plt.show()
  fig.clf()
  plt.close(fig)

  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)
  ax.plot(hw_bins[1:], hw_cum_rec_part[0], label='g1-g1')
  ax.plot(hw_bins[1:], hw_cum_rec_part[1], label='g1-g2')
  ax.plot(hw_bins[1:], hw_cum_rec_part[2], label='g2-g1')
  ax.plot(hw_bins[1:], hw_cum_rec_part[3], label='g2-g2')
  ax.set_xlabel('|w|')
  ax.set_ylabel('cumulative density')    
  ax.legend()
  fig.savefig(os.path.join(fig_dir, 'wm_hist_cum_w_rec_part.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'wm_hist_cum_w_rec_part.pdf'),
              bbox_inches="tight")

  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)
  ax.plot(hw_bins[1:], 1-hw_cum_rec_part[0], label='g1-g1')
  ax.plot(hw_bins[1:], 1-hw_cum_rec_part[1], label='g1-g2')
  ax.plot(hw_bins[1:], 1-hw_cum_rec_part[2], label='g2-g1')
  ax.plot(hw_bins[1:], 1-hw_cum_rec_part[3], label='g2-g2')
  ax.set_xlabel('|w|')  
  ax.set_ylabel('complementary cumulative density')  
  ax.legend()

  ax.set_yscale('log')
  fig.savefig(os.path.join(fig_dir, 'wm_hist_co_cum_w_rec_part_logy.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'wm_hist_co_cum_w_rec_part_logy.pdf'),
              bbox_inches="tight")

  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)
  ax.plot(hw_bins[1:], 1-hw_cum_rec_part[0], label='g1-g1')
  ax.plot(hw_bins[1:], 1-hw_cum_rec_part[1], label='g1-g2')
  ax.plot(hw_bins[1:], 1-hw_cum_rec_part[2], label='g2-g1')
  ax.plot(hw_bins[1:], 1-hw_cum_rec_part[3], label='g2-g2')
  ax.set_xlabel('|w|')
  ax.set_ylabel('complementary cumulative density')
  ax.legend()
  ax.set_xscale('log')
  ax.set_yscale('log')
  fig.savefig(os.path.join(fig_dir, 'wm_hist_co_cum_w_rec_part_logy.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'wm_hist_co_cum_w_rec_part_logy.pdf'),
              bbox_inches="tight")

  plt.show()
  fig.clf()
  plt.close(fig)


def make_symmetric_positive_matrix(A):
  A_syn = np.abs(A - np.diag(np.diag(A)))  # 対角成分を0に, 全体を絶対値に
  A_syn = A_syn + A_syn.T  # 対称成分の和をとって対称行列にする
  return A_syn


def make_group_list(gp):
  ''' returns group list and is_same_group matrix
    returns:
      groups: (N,) array, 0 if i is in group 1, 1 if i is in group 2
      group_list: a list of lists containing index of group members
      is_same_group: (N, N) matrix, 1 if i and j are in the same group, 0 otherwise
  '''
  groups = np.zeros(gp.N, dtype=int)
  groups[gp.g2lim[0]:gp.g2lim[1]] = 1
  group_list = [[], []]
  for i in range(gp.g1lim[0], gp.g1lim[1]):
    group_list[0].append(i)
  for i in range(gp.g2lim[0], gp.g2lim[1]):
    group_list[1].append(i)

  gt = np.reshape(groups, (gp.N, 1))
  is_same_group = (gt.T == gt)
  return groups, group_list, is_same_group


def separation_target(W_out, group_list):
  ''' calculate separation measure of output weights for two groups and two targets
  Args:
    W_out: (N, N_out)
    group_list: a list of lists containing index of group members
  Returns:
    D_target: separation measure
    d_t: difference of weights for two targets. shape: (N,)
  '''

  nout = W_out.shape[1]//2
  aw_t1 = np.sum(np.abs(W_out)[:, 0:nout], axis=1)  # target 1 への重みの絶対値の和
  aw_t2 = np.sum(np.abs(W_out)[:, nout:2*nout], axis=1)  # target 2 への重みの絶対値の和
  d_t = (aw_t2 - aw_t1)  # difference　shape: (N,)
  sum_aw_t = np.sum(aw_t1 + aw_t2)
  D_target = np.abs(
      (np.sum(d_t[group_list[0]]) - np.sum(d_t[group_list[1]])))
  D_target /= sum_aw_t

  return D_target, d_t


def separation_input(W_in, group_list):
  ''' calculate separation measure of input weights for two groups and two targets
  Args:
    W_in: (N_in, N)
    group_list: a list of lists containing index of group members
  Returns:
    D_in: separation measure
    d_in: difference of weights for two targets. shape: (N,)
  '''
  #W_in: (nin*2, N)
  nin = W_in.shape[0]//2
  aw_in1 = np.sum(np.abs(W_in)[0:nin, :], axis=0)  # input 1 からの重みの和
  aw_in2 = np.sum(np.abs(W_in)[nin:2*nin, :], axis=0)  # input 2 からの重みの和
  d_in = (aw_in2 - aw_in1)  # difference
  sum_aw_in = np.sum(aw_in1 + aw_in2)
  D_in = np.abs((np.sum(d_in[group_list[0]]) - np.sum(d_in[group_list[1]])))
  D_in /= sum_aw_in

  return D_in, d_in

# modularity


def modularity_weighted(A, group_list):
  """ calculating modularity (Q) by eq. 7.76 in Newman's book

  Args:
      A (2dim array): Adjacency matrix (assuming symmetry)
      group_list (array): a list of lists containing index of group members

  Returns:
      Q: modularity measure
      e_ij: connection between i-th and j-th groups
      a_i: total vertex in i

  """
  N = A.shape[0]
  M = np.sum(A)/2.0

  n_group = len(group_list)
  e_ij = np.zeros((n_group, n_group))
  for ii, g_i in enumerate(group_list):
    for jj, g_j in enumerate(group_list):
      for kk in g_i:
        for ll in g_j:
          e_ij[ii, jj] += A[kk, ll]
  e_ij /= 2*M

  a_i = np.sum(e_ij, axis=1)

  Q = 0
  for ii in range(n_group):
    Q += e_ij[ii, ii] - a_i[ii]**2

  return Q, e_ij, a_i


def Q_weight_straight(A, groups):
  """ calculating modularity Q by eq. 7.69
    The result should be the same as modularity_weighted.
  """

  gt = np.reshape(groups, (len(groups), 1))
  is_same_group = (gt.T == gt)
  print(is_same_group)
  K = np.sum(A, axis=1)  # degree of each node
  m = np.sum(A)/2  # total number of edge
  sum_same_group = np.sum(A*is_same_group)
  Q = 0.5*np.sum((A - np.outer(K, K)/(2*m)) * is_same_group)/m
  Qmax = 1 - (0.5/m)*np.sum((np.outer(K, K)/(2*m)) * is_same_group)

  return Q, K, Qmax, is_same_group


def calc_modularity_and_separation(model, task, w_recorder, gp, groups, group_list):
  ''' run simulation and calculate modularity and separation measure for each recorded weights.
  Args:
    model: RNN model
    task: task
    w_recorder: weight recorder
    gp: global parameters
    groups: (N,) array, 0 if i is in group 1, 1 if i is in group 2
    group_list: a list of lists containing index of group members
  Returns:
    Q_strs: (n_steps,) array, modularity of structural connectivity
  '''

  Q_strs = []
  Q_corrs = []
  Q_nom_strs = []
  Q_nom_corrs = []

  D_outs = []
  D_ins = []
  D_corrs = []

  for ind, steps in enumerate(w_recorder.steps):
    w_recorder.restore_weights(model, ind=ind)  # 重み読み込み
    #run
    [input, target], [input_stack, target_signal_stack], [
        z_tr, x_tr], [z, r1, r2] = _run_once(task, model, gp)
    #correlation
    corr_X, corr_targ_X = corr_x_and_target(
        model.x.numpy(), target_signal_stack[0].numpy())

    W_in, W_rec, W_out, W_fb = weights_to_numpy(model)
    #W_tot = get_W_total(model, gp)
    W_rec_syn = make_symmetric_positive_matrix(W_rec)

    # Q_str, e_ij, a_i = modularity_weighted(W_tot_syn, group_list)
    Q_str, K, Q_str_max, is_same_group = Q_weight_straight(W_rec_syn, groups)
    Q_nom_str = Q_str/Q_str_max

    # G = nx.from_numpy_array(W_tot_syn)
    # Q_str2 = nx.algorithms.community.quality.modularity(G,group_list)
    print(f'modularity Q_str = {Q_str:.5}')

    corr_X_syn = make_symmetric_positive_matrix(corr_X)
    # Q_corr, e_ij, a_i = modularity_weighted(corr_X_syn, group_list)
    Q_corr, K, Q_corr_max, is_same_group = Q_weight_straight(
        corr_X_syn, groups)
    Q_nom_corr = Q_corr/Q_corr_max

    # G_corr = nx.from_numpy_array(corr_X_syn)
    # Q_corr2 = nx.algorithms.community.quality.modularity(G_corr, group_list)
    print(f'modularity Q_corr = {Q_corr:.5}')
    Q_strs.append(Q_str)
    Q_corrs.append(Q_corr)
    Q_nom_strs.append(Q_nom_str)
    Q_nom_corrs.append(Q_nom_corr)

    # separation measure

    D_t, d_t = separation_target(W_out, group_list)
    D_in, d_in = separation_input(W_in, group_list)
    D_corr, d_corr = separation_target(corr_targ_X.T, group_list)

    D_outs.append(D_t)
    D_ins.append(D_in)
    D_corrs.append(D_corr)

  return Q_strs, Q_corrs, Q_nom_strs, Q_nom_corrs, D_outs, D_ins, D_corrs


def make_fig_Q_and_separation_steps(w_recorder, Q_strs, Q_corrs, Q_nom_strs, Q_nom_corrs, D_ins, D_outs, D_corrs, fig_dir):
  ''' make figure for changes of modularity and separation measure against training steps.
  Args:
    w_recorder: weight recorder 
    Q_strs: (n_steps,) array, modularity of structural connectivity
    Q_corrs: (n_steps,) array, modularity of functional connectivity
    Q_nom_strs: (n_steps,) array, normalized modularity of structural connectivity
    Q_nom_corrs: (n_steps,) array, normalized modularity of functional connectivity
    D_ins: (n_steps,) array, separation measure of input weights
    D_outs: (n_steps,) array, separation measure of output weights
    D_corrs: (n_steps,) array, separation measure of correlation matrix
    fig_dir: str, directory to save figures
  '''
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)
  ax.plot(w_recorder.steps, Q_strs, label='$Q_{str}$', color='blue')
  ax.plot(w_recorder.steps, Q_corrs, label='$Q_{corr}$', color='red')
  ax.set_xlabel('training steps')
  ax.set_ylabel('modularity')
  ax.legend()

  fig.savefig(os.path.join(fig_dir, 'q_steps.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'q_steps.pdf'),
              bbox_inches="tight")
  plt.show()
  fig.clf()
  plt.close(fig)

  #
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)
  ax.plot(w_recorder.steps, Q_nom_strs, label='$Q_{str}$', color='blue')
  ax.plot(w_recorder.steps, Q_nom_corrs, label='$Q_{corr}$', color='red')
  ax.set_xlabel('training steps')
  ax.set_ylabel('normalized modularity')
  ax.legend()

  fig.savefig(os.path.join(fig_dir, 'q_nom_steps.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'q_nom_steps.pdf'),
              bbox_inches="tight")
  plt.show()
  fig.clf()
  plt.close(fig)

  #

  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111)
  ax.plot(w_recorder.steps, D_ins, label='$D_{in}$', color='#4daf4a')
  ax.plot(w_recorder.steps, D_outs, label='$D_{out}$', color='#984ea3')
  ax.plot(w_recorder.steps, D_corrs, label='$D_{corr}$', color='red')
  ax.set_xlabel('training steps')
  ax.set_ylabel('separation index')
  ax.legend()

  fig.savefig(os.path.join(fig_dir, 'separation.png'),
              bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'separation.pdf'),
              bbox_inches="tight")
  plt.show()
  fig.clf()
  plt.close(fig)


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
  #%% setting task
  from wmtask_train import TASK_NAME
  task_name = TASK_NAME
  checkpoint_dir = './checkpoints'

  if task_name == 'wm':
    import gp_wm as gp
  elif task_name == 'mfs':
    import gp_mfs as gp
  elif task_name == 'rossler':
    import gp_rossler as gp
  else:
    import gp_rossler as gp # dummy task    

  if gp.task_name == 'wm':
    data_dir = './data/wm'
  elif gp.task_name == 'mfs':
    data_dir = './data/mfs'
  elif gp.task_name == 'rossler':
    data_dir = './data/rossler'
  else:
    data_dir = './data/periodic'

  plt.rcParams["font.size"] = 16

  tf.config.experimental.enable_tensor_float_32_execution(
      False)  # tf32 (混合精度)を使わない
  tf.keras.backend.set_floatx('float32')

  #%% set recording dictionary
  ana_dic = dict()
  #%% create model and task

  model, task = load_model_and_task(gp)

  # %% Figure setting

  fig_dir, dpi = set_fig_style(gp)

  #%% losses

  # load losses.pkl
  with open(os.path.join(data_dir, 'losses.pkl'), 'rb') as f:
    losses = pickle.load(f)

  make_fig_losses(losses, gp, fig_dir)

  #%% input and target signals, and output of learned model

  make_fig_task(model, task, gp, fig_dir)

  # %% PCA

  pca_component, pca_explained_var_ratio = make_fig_PCA(
      model, task, gp, fig_dir)
  ana_dic.update({'pca_component': pca_component,
                  'pca_explained_var_ratio': pca_explained_var_ratio})
  #%% correlation x-x and x-target

  # Xr = X.reshape((X.shape[0]*X.shape[1], X.shape[2]))  # (bs*t_length, N ))
  # target_r = target_signal_stack[0].numpy().reshape((X.shape[0]*X.shape[1], target.shape[2]))

  corr_X, corr_targ_X, phi_corr, dif_corr = calc_corr_X_target(model, task, gp)

  # Update the analysis dictionary with correlation data
  ana_dic.update({
      'corr_X': corr_X,          # Correlation matrix between RNN states
      # Correlation matrix between RNN states and target signals
      'corr_targ_X': corr_targ_X,
      'phi_corr': phi_corr,      # phi indicate which output is correlated with RNN states
      'dif_corr': dif_corr       # Difference between strength of correlation to two targets
  })

  #%% correlation (un-sorted)

  make_fig_corr_X_targ(corr_X, corr_targ_X, phi_corr, dif_corr, gp, fig_dir, )
  #%% weight matrices

  W_in, W_rec, W_out, W_fb = weights_to_numpy(model)
  std_r = model.r.numpy().reshape((-1, gp.N)).std(axis=0)  # (N,) ユニットごとの出力の標準偏差
  W_rec_nom, W_out_nom = [
      W * std_r.reshape((gp.N, 1)) for W in [W_rec, W_out]]

  #  再帰行列 W-hat は入力側の標準偏差を行列に掛けて正規化したもの
  make_fig_W_rec(W_rec, W_rec_nom, fig_dir)

  #  summarize weights
  make_fig_W_summarize(W_rec, W_out, W_in, fig_dir)

  # Update the analysis dictionary with the weight matrices
  ana_dic.update({
      'W_rec': W_rec,         # Recurrent weight matrix
      'W_rec_out': W_out,     # Output weight matrix
      'W_rec_in': W_in,       # Input weight matrix
      'W_rec_nom': W_rec_nom,  # Normalized recurrent weight matrix
      'W_out_nom': W_out_nom  # Normalized output weight matrix
  })

  #%%   differentiation in W_in, W_out

  d_w_out, corrcoef_dc_d_w_out, _, _ = calc_d_w_out(W_out, W_out_nom, dif_corr)
  d_w_in, corrcoef_dc_d_w_in = calc_d_w_in(W_in, dif_corr, gp.task_name)

  ana_dic.update({
      'd_w_out': d_w_out,
      'd_w_in': d_w_in,
      'corrcoef_dc_d_w_out': corrcoef_dc_d_w_out,
      'corrcoef_dc_d_w_in': corrcoef_dc_d_w_in
  })

  make_fig_diff_out(dif_corr, d_w_out, corrcoef_dc_d_w_out, W_out, fig_dir)
  make_fig_diff_in(dif_corr, d_w_in,  corrcoef_dc_d_w_in, W_in, fig_dir)

  # %% Histograms

  make_fig_weight_hist(W_rec, gp, fig_dir)

  # %% load weight recorder

  with open(os.path.join(data_dir, 'w_recorder.pkl'), 'rb') as f:
    w_recorder = pickle.load(f)
  # %% 重みを詠み込んでシミュレーションを実行してみる．　結果をみてみる．

  if debug_mode:
    for ind, steps in enumerate(w_recorder.steps):
      w_recorder.restore_weights(model, ind=ind)  # model に重みを読み込む関数

      [input, target], [input_stack, target_signal_stack], [
          z_tr, x_tr], [z, r1, r2] = _run_once(task, model, gp)
      plt.plot(z[0])
      plt.show()
      plt.clf()
      plt.close()
  #%% 対象化する．重み付きグラフによりモジュラリティを計算する．

  w_recorder.restore_weights(model, ind=-1)
  W_in, W_rec, W_out, W_fb = weights_to_numpy(model)
  W_rec_syn = make_symmetric_positive_matrix(W_rec)

  # %% グループの定義

  groups, group_list, is_same_group = make_group_list(gp)

  #%% input- and target- separation measure

  if debug_mode:
    D_t, d_t = separation_target(W_out, group_list)  # checking
    D_in, d_in = separation_input(W_in, group_list)
    print(d_t)
    print(d_in)

  # checking two methods returns the same result
  if debug_mode:
    Q, K, Qmax, is_same_group = Q_weight_straight(W_rec_syn, groups)

    Q_2, e_ij, a_i = modularity_weighted(W_rec_syn, group_list)
    print('Q = ', Q)
    print('Q_2 = ', Q_2)

  # checking modularity function
    # structural modularity
    print(f'modularity Q = {Q:.7}')
    Q, K, Qmax, is_same_group = Q_weight_straight(W_rec_syn, groups)
    print(f'modularity Q = {Q:.7}')
    Q_nom = Q/Qmax
    print(f'normalized modularity Q_nom = {Q_nom:.7}')

    # 　相関行列からモジュラリティを計算する．
    #　相関行列
    [input, target], [input_stack, target_signal_stack], [
        z_tr, x_tr], [z, r1, r2] = _run_once(task, model, gp)
    corr_X, corr_targ_X = corr_x_and_target(
        model.x.numpy(), target_signal_stack[0].numpy())

    corr_X_syn = make_symmetric_positive_matrix(corr_X)

    Q_corr, K, Q_corr_max, is_same_group = Q_weight_straight(
        corr_X_syn, groups)
    Q_corr_nom = Q_corr/Q_corr_max
    # G_corr = nx.from_numpy_array(corr_X_syn)
    # Q_corr = nx.algorithms.community.quality.modularity(G_corr, group_list)
    print(f'modularity Q_corr = {Q_corr:.7}')
    print(f'normalized modularity Q_corr_nom = {Q_corr_nom:.7}')

  # %%　modularity and separation measure for each step

  Q_strs, Q_corrs, Q_nom_strs, Q_nom_corrs, D_outs, D_ins, D_corrs = calc_modularity_and_separation(
      model, task, w_recorder, gp, groups, group_list)

  # Store modularity and separation measures in a dictionary
  ana_dic.update({
      'steps_q': w_recorder.steps, # used for plotting x-axis for dev of Qs
      'Q_strs': Q_strs,          # Modularity of the weight matrix
      'Q_corrs': Q_corrs,        # Modularity of the correlation matrix
      'Q_nom_strs': Q_nom_strs,  # Normalized modularity of the weighted graph
      'Q_nom_corrs': Q_nom_corrs,  # Normalized modularity of the correlation matrix
      'D_outs': D_outs,          # Output separation measure
      'D_ins': D_ins,            # Input separation measure
      'D_corrs': D_corrs,        # Correlation separation measure
      'Q_str_last': Q_strs[-1],          # Modularity of the weight matrix
      'Q_corr_last': Q_corrs[-1],        # Modularity of the correlation matrix
      # Normalized modularity of the weighted graph
      'Q_nom_str_last': Q_nom_strs[-1],
      # Normalized modularity of the correlation matrix
      'Q_nom_corr_last': Q_nom_corrs[-1],
      'D_out_last': D_outs[-1],          # Output separation measure
      'D_in_last': D_ins[-1],            # Input separation measure
      'D_corr_last': D_corrs[-1]         # Correlation separation measure
  })

  make_fig_Q_and_separation_steps(
      w_recorder, Q_strs, Q_corrs, Q_nom_strs, Q_nom_corrs, D_ins, D_outs, D_corrs, fig_dir)

  # %% save ana_dict using pickle

  with open(os.path.join(data_dir, 'ana_dict.pkl'), 'wb') as f:
    pickle.dump(ana_dic, f)


  #%% sorting test

  cor_sort_index, dif_corr_sorted, corr_xx_sorted, corr_targ_sorted = corr_sorting(
      corr_X, corr_targ_X)
  # 降順ソートできる
  dif_corr[cor_sort_index]

  #%% 重みのソートをする
  def sort_model_weight(model, sort_index):
    W_rec = model.W_rec.numpy()
    W_rec_sorted = W_rec[sort_index, :][:, sort_index]

    W_in = model.W_in.numpy()
    W_in_sorted = W_in[:, sort_index]

    W_out = model.W_out.numpy()
    W_out_sorted = W_out[sort_index, :]
    W_fb = model.W_fb.numpy()
    W_fb_sorted = W_fb[:,sort_index]
    # bias term
    bias = model.bias.numpy()
    bias_sorted = bias[sort_index]

    model.W_rec.assign(W_rec_sorted)
    model.W_in.assign(W_in_sorted)

    model.W_out.assign(W_out_sorted)
    model.bias.assign(bias_sorted)
    model.W_fb.assign(W_fb_sorted)

  # %%
  # run をして相関行列を作る．　ソートした後にまた相関行列を作る．　比較する

  [input, target], [input_stack, target_signal_stack], [
      z_tr, x_tr], [z, r1, r2] = _run_once(task, model, gp)
  X = model.x.numpy()
  corr_X, corr_targ_X = corr_x_and_target(X, target_signal_stack[0].numpy())
  cor_sort_index, dif_corr_sorted, corr_xx_sorted, corr_targ_sorted = corr_sorting(
      corr_X, corr_targ_X)

  plt.imshow(np.abs(corr_X))
  plt.show()
  plt.plot(np.abs(corr_targ_X.T))

  #%%
  sort_model_weight(model, cor_sort_index)
  # %%

  [input, target], [input_stack, target_signal_stack], [
      z_tr, x_tr], [z, r1, r2] = _run_once(task, model, gp)
  X = model.x.numpy()
  corr_X, corr_targ_X = corr_x_and_target(X, target_signal_stack[0].numpy())
  cor_sort_index, dif_corr_sorted, corr_xx_sorted, corr_targ_sorted = corr_sorting(
      corr_X, corr_targ_X)

  plt.imshow(np.abs(corr_X))
  plt.show()
  plt.plot(np.abs(corr_targ_X.T))




# %%
