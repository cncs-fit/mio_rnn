#%% generate lorenz and rossler time series and save

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random 
import chaos
import os
# from myMINE_ox import *


#%%

# hyperperameters

time_len_l = 100 
time_len_r = 700
validation_split_rate=0
batch_size = 50
data_size = 5000
wave_length = 14000
n_epochs = 120 #
# n_epochs = 2
dir_name = 'Chaos_Signals'
os.makedirs(dir_name, exist_ok=True)

#%%

def generate_random_matrix(row, column=3):
  return np.random.uniform(0, 1, size=(row, column))

lorenz = chaos.Lorenz(sigma=10.0, beta=8.0/3.0, rho=28.0)
rossler = chaos.Rossler(a=0.2, b=0.2, c=5.7)

def chaos_gene(data_size):
  lorenz_chaos = chaos.ChaosGenerator(0.0, time_len_l, wave_length, function=lorenz)
  rossler_chaos = chaos.ChaosGenerator(0.0, time_len_r, wave_length, function=rossler)
  init_list1 = generate_random_matrix(data_size)
  init_list2 = generate_random_matrix(data_size)
  _, result1 = np.array(lorenz_chaos.calculate(init_list1))
  _, result2 = np.array(rossler_chaos.calculate(init_list2))
  return result1[:,4000:,:], result2[:,4000:,:]

#%% generate lorenz and rossler time series and save

data1, data2 = chaos_gene(data_size)
for i in range(n_epochs):
  data1, data2 = chaos_gene(data_size)
  np.save('Chaos_Signals/data1_' + str(i+1), data1)
  np.save('Chaos_Signals/data2_' + str(i+1), data2)
  print(f"Generated and saved data1_{i+1} and data2_{i+1}")

# %%
plt.plot(data1[0,:1000,0], label='Lorenz X')
plt.plot(data2[0,:1000,0], label='Rossler X')
plt.savefig('Chaos_Signals/lorenz_rossler_example.png')
plt.show()

# %%
