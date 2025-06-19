
# 複数回学習実験を試行するためのスクリプト

# パラメータを設定する
# 記録用のdataFrameを用意する
# 1回分の試行する関数に与えるパラメータはgpとseed, デバッグモード可否
# 複数回試行するループに入る
#  試行
#  分析
#  記録
#  元データファイル保存

# TODO: タスク切り替えをargparseで行う．


#%%

import wm_analysis_essence as ana
from tools import clean_saved_directories
from run_once import one_trial
import pandas as pd
from numpy import linalg as LA  # linear algebra
import time
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'


# parameters
N_trial = 20
SEED0 = 1001
TASK_NAME = 'wm'


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
    from IPython import get_ipython #type: ignore
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
  if TASK_NAME == 'wm':
    import gp_wm as gp
  elif TASK_NAME == 'mfs':
    import gp_mfs as gp
  elif TASK_NAME == 'rossler':
    import gp_rossler as gp
  else:
    print('task_name is not valid')
    exit()

  if gp.task_name == 'wm':
    data_dir = './data/wm'
  elif gp.task_name == 'mfs':
    data_dir = './data/mfs'
  elif gp.task_name == 'rossler':
    data_dir = './data/rossler'
  else:
    print('task_name is not valid')
    exit()

  multiple_trials_base_dir = './multiple_trials'
  multiple_trials_dir = os.path.join(
      multiple_trials_base_dir, gp.experiment_name)

  #clean up directory of checkpoint, data, figure, and multiple_trials
  clean_saved_directories()

  # DataFrame を作る
  # index は試行Number
  # 記録するデータ: task-loss, mi-estimation, modularity(2), D_in, D_out, D_corr
  df = pd.DataFrame(index=range(N_trial), columns=[
                    'task-loss', 'mi', 'Q_str', 'Q_corr', 'D_in', 'D_out', 'D_corr'])

  # gp.epochs = 100 # modify epochs for debug
  os.makedirs(multiple_trials_dir, exist_ok=True)

  for nt in range(N_trial):
    seed = SEED0 + nt
    print('-------------------------')
    print('-------------------------')
    print('trial number: ', nt)
    print('-------------------------')
    print('-------------------------')

    model, task, losses, w_recorder = one_trial(
        seed=seed, gp=gp, debug_mode=False)

    # perform main analysis and make figure in wm_analysis_essence
    ana_dic = dict()
    fig_dir, dpi = ana.set_fig_style(gp)
    ana.make_fig_losses(losses, gp, fig_dir)  # save fig of losses
    ana.make_fig_task(model, task, gp, fig_dir)
    pca_component, pca_explained_var_ratio = ana.make_fig_PCA(
        model, task, gp, fig_dir)
    corr_X, corr_targ_X, phi_corr, dif_corr = ana.calc_corr_X_target(
        model, task, gp)
    ana.make_fig_corr_X_targ(
        corr_X, corr_targ_X, phi_corr, dif_corr, gp, fig_dir)
    W_in, W_rec, W_out, W_fb = ana.weights_to_numpy(model)
    std_r = model.r.numpy().reshape((-1, gp.N)).std(axis=0)  # (N,) ユニットごとの出力の標準偏差
    W_rec_nom, W_out_nom = [
        W * std_r.reshape((gp.N, 1)) for W in [W_rec, W_out]]
    ana.make_fig_W_rec(W_rec, W_rec_nom, fig_dir)
    ana.make_fig_W_summarize(W_rec, W_out, W_in, fig_dir)

    d_w_out, corrcoef_dc_d_w_out, _, _ = ana.calc_d_w_out(
        W_out, W_out_nom, dif_corr)
    d_w_in, corrcoef_dc_d_w_in = ana.calc_d_w_in(W_in, dif_corr, gp.task_name)

    ana.make_fig_diff_out(
        dif_corr, d_w_out, corrcoef_dc_d_w_out,  W_out, fig_dir)
    ana.make_fig_diff_in(dif_corr, d_w_in, corrcoef_dc_d_w_in, W_in, fig_dir)
    ana.make_fig_weight_hist(W_rec, gp, fig_dir)
    groups, group_list, is_same_group = ana.make_group_list(gp)

    Q_strs, Q_corrs, Q_nom_strs, Q_nom_corrs, D_outs, D_ins, D_corrs = ana.calc_modularity_and_separation(
        model, task, w_recorder, gp, groups, group_list)

    ana.make_fig_Q_and_separation_steps(w_recorder, Q_strs, Q_corrs, Q_nom_strs, Q_nom_corrs,
                                        D_ins, D_outs, D_corrs, fig_dir)

    # store results to ana_dic

    ana_dic.update({
        'pca_component': pca_component,
        'pca_explained_var_ratio': pca_explained_var_ratio,
        'corr_X': corr_X,
        'corr_targ_X': corr_targ_X,
        'phi_corr': phi_corr,
        'dif_corr': dif_corr,
        'W_in': W_in,
        'W_rec': W_rec,
        'W_out': W_out,
        'W_fb': W_fb,
        'std_r': std_r,
        'W_rec_nom': W_rec_nom,
        'W_out_nom': W_out_nom,
        'd_w_out': d_w_out,
        'corrcoef_dc_d_w_out': corrcoef_dc_d_w_out,
        'd_w_in': d_w_in,
        'corrcoef_dc_d_w_in': corrcoef_dc_d_w_in,
        'groups': groups,
        'group_list': group_list,
        'steps_q': w_recorder.steps,
        'Q_strs': Q_strs,
        'Q_corrs': Q_corrs,
        'Q_nom_strs': Q_nom_strs,
        'Q_nom_corrs': Q_nom_corrs,
        'D_outs': D_outs,
        'D_ins': D_ins,
        'D_corrs': D_corrs,
        'Q_str_last': Q_strs[-1],          # Modularity of the weight matrix
        # Modularity of the correlation matrix
        'Q_corr_last': Q_corrs[-1],
        'Q_nom_str_last': Q_nom_strs[-1],
        'Q_nom_corr_last': Q_nom_corrs[-1],
        'D_out_last': D_outs[-1],          # Output separation measure
        'D_in_last': D_ins[-1],            # Input separation measure
        'D_corr_last': D_corrs[-1]         # Correlation separation measure
    })
    # save ana_dic
    with open(os.path.join(data_dir, 'ana_dic.pkl'), 'wb') as f:
      pickle.dump(ana_dic, f)

    # store main data to df
    df.loc[nt, 'task-loss'] = losses['task'][-1]
    df.loc[nt, 'mi'] = losses['mi_before'][-1]
    if gp.task_name == 'wm':
      df.loc[nt, 'acc'] = losses['accs'][-1]
      df.loc[nt, 'acc'] = losses['accs'][-1]
    df.loc[nt, 'Q_str'] = ana_dic['Q_str_last']
    df.loc[nt, 'Q_corr'] = ana_dic['Q_corr_last']
    df.loc[nt, 'Q_nom_str'] = ana_dic['Q_nom_str_last']
    df.loc[nt, 'Q_nom_corr'] = ana_dic['Q_nom_corr_last']
    df.loc[nt, 'D_in'] = ana_dic['D_in_last']
    df.loc[nt, 'D_out'] = ana_dic['D_out_last']
    df.loc[nt, 'D_corr'] = ana_dic['D_corr_last']

    # save df
    print(df)
    df.to_csv(os.path.join(data_dir, 'df.csv'))

    # copy checkpoint_dir, data_dir, fig_dir to trial_dir using unix command
    save_dir = os.path.join(multiple_trials_dir, 'trial' + str(nt))

    os.makedirs(save_dir, exist_ok=True)
    os.system('cp -r ' + gp.checkpoint_dir + ' ' + save_dir)
    os.system('cp -r ' + data_dir + ' ' + save_dir)
    os.system('cp -r ' + fig_dir + ' ' + save_dir)

  #save df
  df.to_csv(os.path.join(multiple_trials_dir, 'df.csv'), float_format='%.5f')
# %%
