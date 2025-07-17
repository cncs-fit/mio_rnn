#%% analyze a single run

import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import random 
from sklearn.preprocessing import StandardScaler
import chaos
from myMINE_ox import *
from utils import calc_Q_and_D, WeightRecorder, r2_score, calcQ_from_weight, community_analysis
from model import MyModel
      
# %%

# hyperparameters
#TODO sharing hyperparameters between files
time_len_l = 100
time_len_w = 700
validation_split_rate=0
batch_size = 50
data_size = 1000
wave_length = 14000


train_size = int(data_size * (1 - validation_split_rate))
n_steps = train_size // batch_size

lambda_I = 0.005 # regularization parameter for MINE
lambda_L2 = 0.001 # L2 regularization parameter for GRU
lambda_L2_output = 0.001 # L2 regularization parameter for GRU
s_mine = 0.05 # noise strength for MINE training
N = 100 # number of nodes in the hidden layer
step_per_epoch = data_size // batch_size

SEED0 = 625

data_dir = 'data'
chaos_signals_dir = 'Chaos_Signals'
checkpoint_dir = 'checkpoints'
os.makedirs(data_dir, exist_ok=True)
os.makedirs(chaos_signals_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

def set_rand_seed(rand_seed=1):
    tf.random.set_seed(seed=rand_seed)
    np.random.seed(seed=rand_seed+1)



# %%

def generate_random_matrix(row ,column=3):
    return[[random.uniform(0, 1) for _ in range(column)] for _ in range(row)]


def random_batch(X, y, batch_size):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]


#%%


def analyze_results(model, 
                    mymine,
                    stats,
                    x_test=None, y_test=None,
                    batch_size=50,
                    N=100,
                    data_dir='data',                    
                    ):
  set_rand_seed(SEED0)

  # MINE
  mymine = MINE_calculator()
  mymine.init_m()    

  node1 = {str(i+1) for i in range(N//2)}
  node2 = {str(i+1) for i in range(N//2, N)}

  # loading test dataset 
  if x_test is None or y_test is None:
    data1_test = np.load('Chaos_Signals/data1_120.npy')
    data2_test = np.load('Chaos_Signals/data2_120.npy')
    x_test = data1_test + data2_test
    y_test = np.concatenate([data1_test, data2_test], 2)

  x_test_batch = x_test[:batch_size]
  y_test_batch = y_test[:batch_size]

  g1 = tf.random.get_global_generator()
  g2 = tf.random.get_global_generator()
  
  loss_fn = keras.losses.mean_squared_error

  # 

  @tf.function
  def test_step(x_test_batch, y_test_batch):  
    y_pred = model(x_test_batch)

    x1 = model.x[:,:,:N//2] + g1.normal(shape=(batch_size,10000,N//2)) * s_mine #type: ignore
    x2 = model.x[:,:,N//2:] + g2.normal(shape=(batch_size,10000,N//2)) * s_mine #type: ignore
    mival, T = mymine(x1,x2)

    test_loss = tf.reduce_mean(loss_fn(y_test_batch[:,200:,:], y_pred[:,200:,:]))  

    return test_loss, mival, y_pred


  
  test_loss, mival, y_pred_test = test_step(x_test_batch, y_test_batch) # type:ignore
  # print("Test loss:", test_loss.numpy())
  

  # figures
  fig_dir = 'figures'
  os.makedirs(fig_dir, exist_ok=True)
  # default font size
  plt.rcParams.update({'font.size': 24})
  # default tic font size
  plt.rcParams.update({'xtick.labelsize': 18, 'ytick.labelsize': 18})

  # plot losses 
  fig, ax = plt.subplots(1, 1, figsize=(10, 5))
  ax.plot(np.arange(len(stats['losses'])), stats['losses'], label='Loss')
  #ax.plot(np.arange(len(mis)), mis, label='MI')
  ax.set_xlabel(f'Iteration (x{step_per_epoch})')
  #ax.set_ylabel('Value')
  #ax.set_title('Loss and MI over iterations')
  ax.legend()
  fig.savefig(os.path.join(fig_dir, 'losses.png'),bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'losses.pdf'),bbox_inches="tight")  
  plt.show()
  # plot mi

  fig, ax = plt.subplots(1, 1, figsize=(10, 5))
  ax.plot(np.arange(len(stats['MI'])), stats['MI'], label='MI')
  ax.set_xlabel(f'Iteration (x{step_per_epoch})')
  ax.set_ylabel('estimated MI')
  ax.legend()
  fig.savefig(os.path.join(fig_dir, 'mi.png'),bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'mi.pdf'),bbox_inches="tight")
  plt.show()
  #
  # predict vs target, 3 x 2 subplots
  color_predict = 'blue'
  color_target = 'red'
  pl_st = 1000
  pl_ed = 2000
  linestyle_predict = '-'
  linestyle_target = '--'
  fig, axs = plt.subplots(3, 2, figsize=(15, 10))
  # set legend font size
  plt.rcParams.update({'legend.fontsize': 14})
  xyz = ['x', 'y', 'z']
  for i in range(3):
      axs[i, 0].plot(np.arange(pl_st, pl_ed+1),
                    np.array(y_pred_test[0,pl_st:(pl_ed+1),i]), label='Prediction',
                    color=color_predict, linestyle=linestyle_predict)
      axs[i, 0].plot(np.arange(pl_st, pl_ed+1),
                    np.array(y_test_batch[0,pl_st:(pl_ed+1),i]), label='Target',
                    color=color_target, linestyle=linestyle_target)
      axs[i, 0].legend()
      axs[i,0].set_xlabel('Time steps')
      axs[i, 0].set_ylabel(xyz[i])

      axs[i, 1].plot(np.arange(pl_st, pl_ed+1),
                    np.array(y_pred_test[0,pl_st:(pl_ed+1),i+3]), label='prediction',
                    color=color_predict, linestyle=linestyle_predict)
      axs[i, 1].plot(np.arange(pl_st, pl_ed+1),
                    np.array(y_test_batch[0,pl_st:(pl_ed+1),i+3]), label='target',
                    color=color_target, linestyle=linestyle_target)
      axs[i, 1].legend()
      axs[i,1].set_xlabel('Time steps')
      axs[i, 1].set_ylabel(xyz[i])

  fig.savefig(os.path.join(fig_dir, 'pred_vs_target.png'),bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'pred_vs_target.pdf'),bbox_inches="tight")
  plt.show()

  #
  # figures for Q_cors and Q_strs
  plt.rcParams.update({'legend.fontsize': 18})
  fig, ax = plt.subplots(1, 1, figsize=(10, 5))
  ax.plot(np.arange(len(stats['Q_cors'])), stats['Q_cors'], label=r'$Q_{\mathrm{cor}}$',
            color='red', linestyle='solid')
  ax.plot(np.arange(len(stats['Q_strs'])), stats['Q_strs'], label=r'$Q_{\mathrm{str}}$',
            color='blue', linestyle='dashed')
  ax.set_xlabel(f'Iteration (x{step_per_epoch})')
  ax.set_ylabel('Modularity')
  ax.legend()
  fig.savefig(os.path.join(fig_dir, 'Q_cors_Q_strs.png'),bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'Q_cors_Q_strs.pdf'),bbox_inches="tight")

  # figure for D_outs and D_cors
  fig, ax = plt.subplots(1, 1, figsize=(10, 5))
  ax.plot(np.arange(len(stats['D_cors'])), stats['D_cors'], label=r'$D_{\mathrm{cor}}$',
            color='red', linestyle='solid')
  ax.plot(np.arange(len(stats['D_outs'])), stats['D_outs'], label=r'$D_{\mathrm{out}}$',
            color='#984ea3', linestyle='dashed')
  ax.set_xlabel(f'Iteration (x{step_per_epoch})')
  ax.set_ylabel('Separation index')
  ax.legend()
  fig.savefig(os.path.join(fig_dir, 'D_outs_D_cors.png'),bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'D_outs_D_cors.pdf'),bbox_inches="tight")

  plt.show()
  # figure for R2 scores
  labels = ['L x', 'L y', 'L z', 'R x', 'R y', 'R z']
  R2s = np.array(stats['R2s']).transpose()  # shape (6, n_epochs)
  fig, ax = plt.subplots(1, 1, figsize=(10, 5))
  for i in range(6):
      ax.plot(np.arange(R2s.shape[1]), R2s[i], label=labels[i])
  ax.set_xlabel(f'Iteration (x{step_per_epoch})')
  ax.set_ylabel(r'$R^2$')
  ax.legend()
  fig.savefig(os.path.join(fig_dir, 'R2_scores.png'),bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'R2_scores.pdf'),bbox_inches="tight")

                         
  # figure for weight matrix

  # output weights
  a_w_out = np.abs(np.array(model.dense1.get_weights()[0])) # shape (N,6) #type:ignore
  fig, ax = plt.subplots(1, 1, figsize=(8, 3))
  im = ax.imshow(a_w_out.T, cmap='Blues', aspect='auto')
  ax.set_yticks(np.arange(0,6), labels=['Lx', 'Ly', 'Lz', 'Rx', 'Ry', 'Rz'])
  ax.set_xlabel('Neuron index')
  fig.savefig(os.path.join(fig_dir, 'output_weights.png'),bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'output_weights.pdf'),bbox_inches="tight")

  # recurrent weights
  rec_kernel = np.array(model.gru.cell.recurrent_kernel) # shape (N,3*N) #type:ignore
  U = rec_kernel[:,2*N:] # shape (N,N)
  aU = np.abs(U) # absolute value of recurrent weights

  fig, ax = plt.subplots(1, 1, figsize=(8, 8))
  im = ax.imshow(aU, cmap='Blues', aspect='auto')
  # ax.set_title('Recurrent Weights')
  # set aspect ratio to be equal
  ax.set_xlabel('Neuron index')
  ax.set_ylabel('Neuron index')
  ax.set_aspect('equal')
  # fig.colorbar(im, ax=ax, orientation='vertical')
  fig.savefig(os.path.join(fig_dir, 'recurrent_weights.png'),bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'recurrent_weights.pdf'),bbox_inches="tight") 

  # figures for correlation neuron-output
  y_pred = np.array(model(x_test_batch)) # shape (batch_size, time_steps, 6)
  hiddens = np.array(model.x) # shape (batch_size, time_steps, N)

  # calculate correlation coefficients between hidden states and outputs
  cor_h_target = np.zeros((hiddens.shape[0], hiddens.shape[2], y_pred.shape[2])) # shape (batch_size, N, 6)
  for i in range(hiddens.shape[0]):
      for j in range(hiddens.shape[2]):
          for k in range(y_pred.shape[2]):
              cor_h_target[i,j,k] = np.corrcoef(hiddens[i,:,j], y_pred[i,:,k])[0,1] 
  cor_h_target = np.abs(cor_h_target) # absolute value of correlation coefficients
  cor_h_target = np.mean(cor_h_target, axis=0) # shape (N, 6)
  m_cor_h_target1 = np.mean(cor_h_target[:,0:3], axis=1) # mean correlation for target1
  m_cor_h_target2 = np.mean(cor_h_target[:,3:], axis=1) # mean correlation for target2
  # plot correlation coefficients

  fig, ax = plt.subplots(1, 1, figsize=(8, 8))
  ax.plot(m_cor_h_target1[0:N//2], m_cor_h_target2[0:N//2],'o', color='steelblue', markersize=5, label='Group 1')
  ax.plot(m_cor_h_target1[N//2:], m_cor_h_target2[N//2:],'^', color='tomato', markersize=5, label='Group 2')
  # set aspect ratio to be equal
  ax.set_aspect('equal')
  # Make sure limits are the same on both axes for a perfect square
  limits = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
  ax.set_xlim(limits) # type:ignore
  ax.set_ylim(limits) #type:ignore
  ax.legend()
  ax.set_xlabel('Mean correlation with target 1')
  ax.set_ylabel('Mean correlation with target 2')
  #ax.set_title('Neuron-Target Correlation')
  fig.savefig(os.path.join(fig_dir, 'neuron_target_correlation.png'),bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'neuron_target_correlation.pdf'),bbox_inches="tight")

  plt.show()

  # correlation matrix between hidden states

  cor_matrix = np.zeros((hiddens.shape[0], hiddens.shape[2], hiddens.shape[2])) # shape (batch_size, N, N)
  for i in range(hiddens.shape[0]):
      cor_matrix[i] = np.corrcoef(hiddens[i].T)
  cor_matrix = np.mean(cor_matrix, axis=0) # shape (N, N)
  a_cor_matrix = np.abs(cor_matrix) # absolute value of correlation coefficients

  fig, ax = plt.subplots(1, 1, figsize=(8, 8))
  im = ax.imshow(a_cor_matrix, cmap='Reds', aspect='auto')
  # set color bar
  fig.colorbar(im, ax=ax, orientation='vertical')
  # ax.set_title('Correlation Matrix of Hidden States')
  ax.set_aspect('equal')
  ax.set_xlabel('Neuron index')
  ax.set_ylabel('Neuron index')
  # save the figure
  fig.savefig(os.path.join(fig_dir, 'correlation_matrix.png'),bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'correlation_matrix.pdf'),bbox_inches="tight")
 
  plt.show()  

  # community analysis
  sim_nmi, fig = community_analysis(a_cor_matrix)
  fig.savefig(os.path.join(fig_dir, 'community_analysis.png'),bbox_inches="tight")
  fig.savefig(os.path.join(fig_dir, 'community_analysis.pdf'),bbox_inches="tight")

  # add sim_nmi to stats
  stats['sim_nmi'] = [sim_nmi]
  # community analysis of weight U 
  aU_sym = (aU + aU.T) / 2  # make the matrix symmetric
  # calculate similarity and nmi
  sim_nmi_U, fig_U = community_analysis(aU_sym)
  fig_U.savefig(os.path.join(fig_dir, 'community_analysis_U.png'),bbox_inches="tight")
  fig_U.savefig(os.path.join(fig_dir, 'community_analysis_U.pdf'),bbox_inches="tight")
  # add sim_nmi_U to stats
  stats['sim_nmi_U'] = [sim_nmi_U]


  #   calculate Q for resetting gate and update gate

  Urec = np.array(model.gru.cell.recurrent_kernel) # shape (N,3*N)

  
  Uz = Urec[:,0:N] # shape (N,N)
  Ur = Urec[:,N:2*N] # shape (N,N)
  U = Urec[:,2*N:] # shape (N,N)

  node1 = [i for i in range(N//2)]
  node2 = [i for i in range(N//2, N)]         
          
  Q_str = calcQ_from_weight(U, node1, node2) # type:ignore
  Q_str_z = calcQ_from_weight(Uz, node1, node2) # type:ignore
  Q_str_r = calcQ_from_weight(Ur, node1, node2) # type:ignore

  print("Q_str:", Q_str
        , "Q_str_r:", Q_str_r, "Q_str_z:", Q_str_z)
  # add Q_str_r and Q_str_z to stats

  stats['Q_str_z'] = [Q_str_z]
  stats['Q_str_r'] = [Q_str_r]
  

  # saving stats 
  np.savez(os.path.join(data_dir, 'statistics.npz'), **stats)

  return stats
  # Q_str_var = dict(Q_str=Q_str, Q_str_r=Q_str_r, Q_str_z=Q_str_z)
  # # save the Q_str_var to a file
  # np.savez(os.path.join(data_dir, 'Q_str_var.npz'), **Q_str_var )


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

  model = MyModel(N=N, lambda_L2=lambda_L2)
  # dummy input for model initialization
  dummy_input = np.random.rand(1, 10000, 3).astype(np.float32) 
  model(dummy_input)  # initialize the model
  mymine = MINE_calculator()
  #%%
  # load test data
  data1_test = np.load(os.path.join(chaos_signals_dir, 'data1_120.npy'))
  data2_test = np.load(os.path.join(chaos_signals_dir, 'data2_120.npy'))
  x_test = data1_test + data2_test
  y_test = np.concatenate([data1_test, data2_test], 2)  

  # load the model weights
  save_path = os.path.join(checkpoint_dir, 'ckpt-current')
  model.load_weights(save_path)
  # load the statistics
  stats = dict(np.load(os.path.join(data_dir, 'statistics.npz')))

  # analyze the results
  analyze_results(model, mymine, stats,
                  x_test=x_test, y_test=y_test,
                  batch_size=batch_size,
                  N=N,
                  data_dir=data_dir,
  )
# %%
