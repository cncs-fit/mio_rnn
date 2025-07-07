'''
Run multiple trials (training/analysis) and save results in separate directories
- Repeat n_trials times
- Within the loop, set random seeds, initialize the model, train, and analyze
- Create a multiple_trials directory and under it create directories named 1, 2, ...,n_trials
- After each trial, copy the model (checkpoints directory), data (data directory), and figures (figures directory) to the respective trial directory
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

n_trials = 20
#n_trials = 5

SEED0 = 625
N = 100
batch_size = 50
data_size = 1000
#data_size = 500
n_epochs = 200
lambda_I = 0.005 # regularization parameter for MINE
lambda_L2 = 0.0001 # L2 regularization parameter for GRU
lambda_L2_output = 0.0001 # L2 regularization parameter for output layer
# weaker regularization
# lambda_L2 = 0.00001 # L2 regularization parameter for GRU
# lambda_L2_output = 0.00001 # L2 regularization parameter for output layer

s_mine = 0.05
trial_number_offset = 0

#n_epochs = 4

data_dir = 'data'
figures_dir = 'figures'
checkpoints_dir = 'checkpoints'
os.makedirs(data_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)


multiple_trials_dir = "multiple_trials"
os.makedirs(multiple_trials_dir, exist_ok=True)

def set_rand_seed(rand_seed=1):
    tf.random.set_seed(seed=rand_seed)
    np.random.seed(seed=rand_seed+10000)
    random.seed(seed+10000)


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

  ##### run multiple trials
  for trial in range(trial_number_offset, trial_number_offset + n_trials):
    print(f"Trial {trial+1}/{n_trials+trial_number_offset}")

    # Set random seeds for reproducibility
    seed = SEED0 + trial 
    set_rand_seed(rand_seed=seed)

    # Create a directory for this trial
    trial_dir = os.path.join(multiple_trials_dir, str(trial + 1))
    os.makedirs(trial_dir, exist_ok=True)


    # Train the model
    print(f"lambda_L2: {lambda_L2}, lambda_L2_output: {lambda_L2_output}, lambda_I: {lambda_I}, s_mine: {s_mine}")
    model, mymine, stats, x_test, y_test = training_gru_mine(data_size=data_size, 
                                                  n_epochs=n_epochs, 
                                                  batch_size=batch_size,
                                                  lambda_I=lambda_I,
                                                  lambda_L2=lambda_L2,
                                                  lambda_L2_output=lambda_L2_output,
                                                  s_mine=s_mine,)
    # analyze results
    analyze_results(model, mymine, stats,
                    x_test=x_test, y_test=y_test,
                    batch_size=batch_size,
                    N=N,
                    data_dir=data_dir
    )
    
    # copy data, figures, and checkpoints to the trial directory
    # copyfiles
    os.system('cp -rp ' + checkpoints_dir + ' ' + trial_dir)
    os.system('cp -rp ' + data_dir + ' ' + trial_dir)
    os.system('cp -rp ' + figures_dir + ' ' + trial_dir)




