# analyze the results of multiple runs

#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


TASK_NAME = 'wm'
experiment_name = 'wm_ld0.0creg0.01'
multiple_trial_base_dir = 'multiple_trials'
multiple_trial_dir = os.path.join(multiple_trial_base_dir, experiment_name)

# memo:
# 戦略 accuracy や task-loss で成功の基準を脚きりし，その後 Q_str か Q_corr で分化が起きたかを分類し，条件をクリアしたものを分析する

#%%


if __name__ == '__main__':
  
  # if TASK_NAME == 'wm':
  #   data_dir = 'data/wm'

  if TASK_NAME == 'wm':
    data_dir = 'wm'
  elif TASK_NAME == 'mfs':
    data_dir = 'mfs'
  else:
    print('task_name is not valid')
    exit()
      
  # load dataFrames from csv file
  
  # data_dir = 'multiple_trials/wm_creg0.02' # temporal
  df = pd.read_csv(os.path.join(multiple_trial_dir,'df.csv'), index_col=0)
  df
  #%%
  print(df)

  # %% drawing scatter plots between pairs of Q_str, Q_corr
  var_compare = ['task-loss', 'mi', 'Q_str', 'Q_corr', 'D_in', 'D_out']
  if TASK_NAME == 'wm':
    var_compare.append('acc')
  obj = sns.pairplot(df, vars=var_compare)
  obj.savefig(os.path.join(multiple_trial_dir, 'pairplot.png'))
  

  # %%

  # threshold for task-loss
  if TASK_NAME == 'wm':
    thr_loss = 0.3
  else:
    thr_loss = 0.1

  # threshold for accuracy (wm only)
  thr_acc = 0.9
  # threshold for Q_str
  thr_Q_str = 0.1


  #%% タスク学習が成功かの判定:　extract successful trials
  # 1. task-loss が thr_loss 以下
  # 2. accuracy が thr_acc 以上 (wm only)

  if TASK_NAME == 'wm':
    successful_trials = df[(df['acc'] > thr_acc)]
  else:
    successful_trials = df[(df['task-loss'] < thr_loss)]

  num_successful_trials = successful_trials.shape[0]
  success_rate = num_successful_trials / df.shape[0]
  print(f'success rate: {success_rate:.5}')


  #%% タスク学習が成功した試行のうち，Q_str が thr_Q_str 以上のものを抽出
  suc_differentiated_trials = successful_trials[(successful_trials['Q_str'] > thr_Q_str)]
  num_dif_trials = suc_differentiated_trials.shape[0]
  dif_success_rate = num_dif_trials / df.shape[0]
  print(f'differentiation success rate: {dif_success_rate:.5}')
  suc_differentiated_trials

  #%% それぞれのsuc_differentiatedトライアルのQの発展を ana_dicをロードして集めて，プロットする．
  # Q_corr が Q_str よりも早く上昇することを示す図を作る．
  

  import pickle
  suc_dif_index = suc_differentiated_trials.index

  def collect_Qs_and_Ds(trial_index):
    ana_dics = []  
    Q_strs_list = []
    Q_corrs_list = []
    D_ins_list = []
    D_outs_list = []
    D_corrs_list = []

    for trial in trial_index:
      dir_name = os.path.join(multiple_trial_dir ,'trial' + str(trial))
      print(dir_name)
      # load ana_dic from pickle file
      ana_dic = pickle.load(open(os.path.join(dir_name, data_dir, 'ana_dic.pkl'), 'rb'))    
      ana_dics.append(ana_dic)
      Q_strs_list.append(ana_dic['Q_strs'])
      Q_corrs_list.append(ana_dic['Q_corrs'])
      D_ins_list.append(ana_dic['D_ins'])
      D_outs_list.append(ana_dic['D_outs'])
      D_corrs_list.append(ana_dic['D_corrs'])
      plt.imshow(np.abs(ana_dic['W_rec']), cmap='Blues')
      plt.show()      
      plt.imshow(np.abs(ana_dic['corr_X']), cmap='Reds')
      plt.show()
      epochs_q = ana_dic['steps_q'] #type:ignore temporal       
    Q_strs = np.array(Q_strs_list)
    Q_corrs = np.array(Q_corrs_list)
    D_ins = np.array(D_ins_list)
    D_outs = np.array(D_outs_list)
    D_corrs = np.array(D_corrs_list)


    return Q_strs, Q_corrs, D_ins, D_outs, D_corrs, epochs_q
  
  Q_strs_all, Q_corrs_all, D_ins_all, D_outs_all, D_corrs_all, epochs_q = collect_Qs_and_Ds(df.index.to_numpy())  

  if len(suc_dif_index)>0:
    Q_strs_dif, Q_corrs_dif, D_ins_dif, D_outs_dif, D_corrs_dif, epochs_q = collect_Qs_and_Ds(suc_dif_index)
  # epochs_q = np.arange(0, 500+1, 25) # temporal
  # m_Q_strs_dif = Q_strs_dif.mean(axis=0)
  # m_Q_corrs_dif = Q_corrs_dif.mean(axis=0)  

  # m_D_ins_dif = D_ins_dif.mean(axis=0)
  # m_D_outs_dif = D_outs_dif.mean(axis=0)
  # m_D_corrs_dif = D_corrs_dif.mean(axis=0)


  #%% plot Q_strs and Q_corrs of suc_differentiated_trials and their mean
  from wm_analysis_essence import set_fig_style


  # fig, ax = plt.subplots(figsize=(8, 6))
  # ax.plot(epochs_q, Q_strs_dif.T, color='blue', alpha=0.3)
  # ax.plot(epochs_q, Q_corrs_dif.T, color='red', alpha=0.3)  
  # ax.plot(epochs_q, m_Q_strs_dif, color='blue', alpha=1, linewidth=3, label='Q_str')
  # ax.plot(epochs_q, m_Q_corrs_dif, color='red', alpha=1, linewidth=3, label='Q_corr')
  # ax.legend( fontsize=16)
  # ax.set_xlabel('epoch', fontsize=20)
  # ax.set_ylabel('modularity', fontsize=20)
  # fig.savefig(os.path.join(multiple_trial_dir, 'mean_Q_suc_dif.png'), dpi=300, bbox_inches='tight')
  # fig.savefig(os.path.join(multiple_trial_dir, 'mean_Q_suc_dif.pdf'))  
  # # %% plot D_in, D_out and D_corr of suc_differentiated_trials and their mean
  # fig, ax = plt.subplots(figsize=(8, 6))
  # ax.plot(epochs_q, D_ins_dif.T, color='#4daf4a', alpha=0.3)
  # ax.plot(epochs_q, D_outs_dif.T, color='#984ea3', alpha=0.3)
  # ax.plot(epochs_q, D_corrs_dif.T, color='red', alpha=0.3)  
  # ax.plot(epochs_q, m_D_ins_dif, color='#4daf4a', alpha=1, linewidth=3, label='D_in')
  # ax.plot(epochs_q, m_D_outs_dif, color='#984ea3', alpha=1, linewidth=3, label='D_out')
  # ax.plot(epochs_q, m_D_corrs_dif, color='red', alpha=1, linewidth=3, label='D_corr')
  # ax.legend( fontsize=16)
  # ax.set_xlabel('epoch', fontsize=20)
  # ax.set_ylabel('separation index', fontsize=20)
  # fig.savefig(os.path.join(multiple_trial_dir, 'mean_D_suc_dif.png'), dpi=300, bbox_inches='tight')
  # fig.savefig(os.path.join(multiple_trial_dir, 'mean_D_suc_dif.pdf'))  

  # %%
  def make_fig_Q_steps(epochs_q, Q_strs, Q_corrs, filename ):

    m_Q_strs = Q_strs.mean(axis=0)
    m_Q_corrs = Q_corrs.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs_q, m_Q_corrs, color='red', linestyle='solid', alpha=1, linewidth=3, label='Q_corr')
    ax.plot(epochs_q, m_Q_strs, color='blue',linestyle='dashed', alpha=1, linewidth=3, label='Q_str')

    ax.plot(epochs_q, Q_strs.T, color='blue', linestyle='dashed', alpha=0.3)
    ax.plot(epochs_q, Q_corrs.T, color='red',linestyle='solid', alpha=0.3)  
    ax.legend( fontsize=16)
    ax.set_xlabel('iteration', fontsize=20)
    ax.set_ylabel('modularity', fontsize=20)
    fig.savefig(filename+'.png', dpi=300, bbox_inches='tight')
    fig.savefig(filename+'.pdf', bbox_inches='tight')


  def make_fig_D_steps(epochs_q, D_ins, D_outs, D_corrs, filename ):    
    #  plot D_in, D_out and D_corr of suc_differentiated_trials and their mean
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs_q, D_ins.T, color='#4daf4a', linestyle='dashed',alpha=0.3)
    ax.plot(epochs_q, D_outs.T, color='#984ea3', linestyle='dotted', alpha=0.3)
    ax.plot(epochs_q, D_corrs.T, color='red', linestyle='solid', alpha=0.3)  
    ax.plot(epochs_q, D_corrs.mean(axis=0), color='red', linestyle='solid', alpha=1, linewidth=3, label='D_corr')
    ax.plot(epochs_q, D_ins.mean(axis=0), color='#4daf4a', linestyle='dashed', alpha=1, linewidth=3, label='D_in')
    ax.plot(epochs_q, D_outs.mean(axis=0), color='#984ea3', linestyle='dotted', alpha=1, linewidth=3, label='D_out')
    ax.legend( fontsize=16)
    ax.set_xlabel('iteration', fontsize=20)
    ax.set_ylabel('separation index', fontsize=20)
    fig.savefig(filename+'.png', dpi=300, bbox_inches='tight')
    fig.savefig(filename+'.pdf', bbox_inches='tight')

  # %%
  if len(suc_dif_index)>0:

    q_steps_dif_fname = os.path.join(multiple_trial_dir, 'mean_Q_suc_dif')
    d_steps_dif_fname = os.path.join(multiple_trial_dir, 'mean_D_suc_dif')
    make_fig_Q_steps(epochs_q, Q_strs_dif, Q_corrs_dif, q_steps_dif_fname)
    make_fig_D_steps(epochs_q, D_ins_dif, D_outs_dif, D_corrs_dif, d_steps_dif_fname)
  # %%
  q_steps_all_fname = os.path.join(multiple_trial_dir, 'mean_Q_all')
  d_steps_all_fname = os.path.join(multiple_trial_dir, 'mean_D_all')
  make_fig_Q_steps(epochs_q, Q_strs_all, Q_corrs_all, q_steps_all_fname)
  make_fig_D_steps(epochs_q, D_ins_all, D_outs_all, D_corrs_all, d_steps_all_fname)

# %%
