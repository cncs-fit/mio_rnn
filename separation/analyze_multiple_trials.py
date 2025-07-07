#%%
'''
analyze_multiple_trials.py
analyze the results of multiple trials (training/analysis) and create figures.
- Use the stats data where the results of n_trials are in a folder.
- First, read the stats of each trial and put them into a list.
- Create a pandas DataFrame based on that list to summarize the results.
- Aggregate the final values of Q_cor, Q_str, D_cor, D_out, mi, r2_score, and loss using pandas.
- Create pair plots and other analyses to examine the results.
- For Q and D, create figures showing the time evolution and average values for all trials
- Save the figures in the multiple_trials/figures/ directory.
'''

#%%
import os
import tensorflow as tf
import numpy as np
import random 
from myMINE_ox import *
from model import MyModel
from analyze_results import analyze_results
from gru_training import training_gru_mine
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

n_trials = 20

data_size = 1000
batch_size = 50
n_epochs = 200


data_dir = 'data'
figures_dir = 'figures'
checkpoints_dir = 'checkpoints'
multiple_trials_dir = 'multiple_trials'

os.makedirs(os.path.join(multiple_trials_dir, figures_dir), exist_ok=True)

# %%
if __name__ == "__main__":


  # GPU configuration
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
      tf.config.experimental.set_memory_growth(physical_devices[k], True)
      print('memory growth:', tf.config.experimental.get_memory_growth(
          physical_devices[k]))
  else:
    print("Not enough GPU hardware devices available")


  # load stats files from multiple trials
  stats_list = []
  for i in range(n_trials):
    stats_file = os.path.join(multiple_trials_dir, f'{i+1}', data_dir, 'statistics.npz')
    stats = dict(np.load(stats_file))
    stats_list.append(stats)

  # create a DataFrame from the stats_list
  # use last values  (Q_cors[-1]) of the trials

  df = pd.DataFrame()
  df['trial'] = [i+1 for i in range(n_trials)]
  df['Q_cor'] = [stats['Q_cors'][-1] for stats in stats_list]
  df['Q_str'] = [stats['Q_strs'][-1] for stats in stats_list]
  df['D_cor'] = [stats['D_cors'][-1] for stats in stats_list]
  df['D_out'] = [stats['D_outs'][-1] for stats in stats_list]
  df['MI'] = [stats['MI'][-1] for stats in stats_list]
  df['R2'] = [stats['R2s'][-1] for stats in stats_list] # 6-dim array
  df['loss'] = [stats['losses'][-1] for stats in stats_list]
  df['minR2'] = [np.min(stats['R2s'][-1]) for stats in stats_list]
  # community detection
  df['sim_nmi'] = [stats['sim_nmi'][-1] for stats in stats_list]
  df['sim_nmi_U'] = [stats['sim_nmi_U'][-1] for stats in stats_list]
  # Q_str_r, Q_str_z
  df['Q_str_r'] = [stats['Q_str_r'][-1] for stats in stats_list]
  df['Q_str_z'] = [stats['Q_str_z'][-1] for stats in stats_list]

  df #type:ignore

  #%%
  # stats for R2
  # R2 have 6 components 

  # R2 data to 2 dim (n_trials x 6)
  r2_array = np.array([r2 for r2 in df['R2']])  # shape: (n_trials, 6)

  
  # calculate the mean of R2 for each trial
  r2_trial_means = np.mean(r2_array, axis=1)  # shape: (n_trials,)

    # calculate the mean and std of R2 for each trial
  r2_overall_mean = np.mean(r2_trial_means)
  r2_overall_std = np.std(r2_trial_means)

  # calculate the mean and std of R2 for each component
  r2_component_means = np.mean(r2_array, axis=0)  # shape: (6,)
  r2_component_stds = np.std(r2_array, axis=0)   # shape: (6,)

  
  print(f"Mean of R2 across trials: {r2_overall_mean:.4f}")
  print(f"Standard deviation of R2 across trials: {r2_overall_std:.4f}")
  print(f"Mean of R2 components: {r2_component_means}")
  print(f"Standard deviation of R2 components: {r2_component_stds}")  
                                     

  # add to DataFrame
  df['R2_mean'] = r2_trial_means
  for i in range(6):
    df[f'R2_comp{i+1}'] = r2_array[:, i]

  #%%
  # mean and std of df columns
  df_mean = df.mean()
  df_std = df.std()
  # %%
  df
  #%%
  df_mean
  #%%
  df_std


  #%% aggregate the results and save to a csv file
  df_summary = pd.DataFrame({
      'mean': df_mean,
      'std': df_std
  })  
  df_summary.index.name = 'Metric'
  df_summary.reset_index(inplace=True)
  
  df_summary.to_csv(os.path.join(multiple_trials_dir,  'summary.csv'), index=False)
  # save df
  df.to_csv(os.path.join(multiple_trials_dir, 'df.csv'), index=False)


  #%% pairplot
  # delete R2_comp1, R2_comp2, ..., R2_comp6 
  df_pairplot = df.drop(columns=[f'R2_comp{i+1}' for i in range(6)])
  sns.pairplot(df_pairplot, diag_kind='kde', markers='o')
  # save figure
  plt.savefig(os.path.join(multiple_trials_dir, figures_dir, 'pairplot.png'))


  # %% time course of Q_cor, Q_str, D_cor, D_out 

  # Q_cor の時間発展の平均と標準偏差を計算
  Q_cors = np.array([stats['Q_cors'] for stats in stats_list])
  Q_strs = np.array([stats['Q_strs'] for stats in stats_list ])
  D_cors = np.array([stats['D_cors'] for stats in stats_list  ])
  D_outs = np.array([stats['D_outs'] for stats in stats_list  ])

  # nanを除外して平均を計算
  Q_cors_nonan = np.nan_to_num(Q_cors, nan=np.nanmean(Q_cors))
  m_Qcors = np.nanmean(Q_cors, axis=0)
  std_Qcors = np.nanstd(Q_cors, axis=0)
  Q_strs_nonan = np.nan_to_num(Q_strs, nan=np.nanmean (Q_strs))
  m_Qstrs = np.nanmean(Q_strs, axis=0)
  std_Qstrs = np.nanstd(Q_strs, axis=0)
  D_cors_nonan = np.nan_to_num(D_cors, nan=np.nanmean(D_cors))
  m_Dcors = np.nanmean(D_cors, axis=0)
  std_Dcors = np.nanstd(D_cors, axis=0)
  D_outs_nonan = np.nan_to_num(D_outs, nan=np.nanmean(D_outs))
  m_Douts = np.nanmean(D_outs, axis=0)
  std_Douts = np.nanstd(D_outs, axis=0) 
  # nan の回数を記録と出力
  n_nan_Qcors = np.sum(np.isnan(Q_cors))
  n_nan_Qstrs = np.sum(np.isnan(Q_strs))
  n_nan_Dcors = np.sum(np.isnan(D_cors))
  n_nan_Douts = np.sum(np.isnan(D_outs))
  print(f"Q_cor: {n_nan_Qcors} nan values")
  print(f"Q_str: {n_nan_Qstrs} nan values")
  print(f"D_cor: {n_nan_Dcors} nan values")
  print(f"D_out: {n_nan_Douts} nan values")
                               
                     

  # set font size
  plt.rcParams.update({'font.size': 22})

  # まずQ_cor, Q_strの時間発展をプロット
  fig, ax =plt.subplots(1, 1, figsize=(8,6))
  
  for i in range(n_trials):
    ax.plot(stats_list[i]['Q_cors'], color='red', alpha=0.3)
  ax.plot( m_Qcors,
          label='Mean Q_cor', color='red', linewidth=3, linestyle='solid')
  for i in range(n_trials):
    ax.plot(stats_list[i]['Q_strs'], color='blue',linestyle='dashed', alpha=0.3)
  ax.plot( m_Qstrs,
           label='Mean Q_str', color='blue', linewidth=3, linestyle='dashed')
  ax.set_xlabel(f'iteration ( x{data_size // batch_size} )')
  ax.set_ylabel('Modularity')
  ax.legend()
  plt.savefig(os.path.join(multiple_trials_dir, figures_dir, 'Q_cor_Q_str_time_evolution.png'))
  plt.savefig(os.path.join(multiple_trials_dir, figures_dir, 'Q_cor_Q_str_time_evolution.pdf'))  
  plt.show()  

  # 同様にD_cor, D_outの時間発展をプロット
  fig, ax =plt.subplots(1, 1, figsize=(8,6))
  for i in range(n_trials):
    ax.plot(stats_list[i]['D_cors'], color='red', alpha=0.3)
  ax.plot( m_Dcors,
          label='Mean D_cor', color='red', linewidth=3, linestyle='solid')
  for i in range(n_trials):
    ax.plot(stats_list[i]['D_outs'], color='purple',linestyle='dashed', alpha=0.3)
  ax.plot( m_Douts,
           label='Mean D_out', color='purple', linewidth=3, linestyle='dashed')
  ax.set_xlabel(f'iteration ( x{data_size // batch_size} )')
  ax.set_ylabel('Separation Index')
  ax.legend()
  plt.savefig(os.path.join(multiple_trials_dir, figures_dir, 'D_cor_D_out_time_evolution.png'))
  plt.savefig(os.path.join(multiple_trials_dir, figures_dir, 'D_cor_D_out_time_evolution.pdf'))  
  plt.show()
# %%

# extract trials that satisfy R2_mean > threshold 
R2_threshold = 0.96
high_r2_trials = df[df['R2_mean'] > R2_threshold]
print(f"R2 mean が {R2_threshold} 以上の回:")
print(high_r2_trials)
# %%
