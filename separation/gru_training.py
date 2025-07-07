#%% a main training function to train a GRU model with MINE

import os
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import random 
from sklearn.preprocessing import StandardScaler
from myMINE_ox import *
from utils import calc_Q_and_D, WeightRecorder, r2_score, calcQ_from_weight
from model import MyModel
from analyze_results import analyze_results

      
# %%

N = 100 # number of nodes in the hidden layer
SEED0 = 625
# hyperparameters

n_sm = 20 # number of MINE training steps for one step of GRU training
data_dir = 'data'
chaos_signals_dir = 'Chaos_Signals'
checkpoint_dir = 'checkpoints'
os.makedirs(data_dir, exist_ok=True)
os.makedirs(chaos_signals_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

def set_rand_seed(rand_seed=1):
    tf.random.set_seed(seed=rand_seed)
    np.random.seed(seed=rand_seed+1)


def progress_bar(iteration, total, size=30):
    running = iteration < total
    c = ">" if running else "="
    p = (size - 1) * iteration // total
    fmt = "{{:-{}d}}/{{}} [{{}}]".format(len(str(total)))
    params = [iteration, total, "=" * p + c + "." * (size - p - 1)]
    return fmt.format(*params)

def print_status_bar(iteration, total, loss, metrics=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                          for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print("\r{} - {}".format(progress_bar(iteration, total), metrics), end=end)



# %% data generation
# args_l = (10.0, 28.0, 8.0/3.0)
# args_w = (0.2, 0.2, 5.7)  


def generate_random_matrix(row ,column=3):
    return[[random.uniform(0, 1) for _ in range(column)] for _ in range(row)]

def random_batch(X, y, batch_size):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]


#%%

def training_gru_mine(seed=SEED0,
                      data_size=500, 
                      n_epochs=2, 
                      batch_size=50,
                      lambda_I = 0.005, 
                      lambda_L2 = 0.001, 
                      lambda_L2_output = 0.0001, 
                      s_mine = 0.05, # noise strength for MINE training
):

  optimizer = Adam(learning_rate=0.001)
  loss_fn = keras.losses.mean_squared_error
  mean_loss = keras.metrics.Mean()
  metrics = [keras.metrics.MeanAbsoluteError()]
  mi_mean = keras.metrics.Mean(name='mi')

  model = MyModel(lambda_L2=lambda_L2, lambda_L2_output=lambda_L2_output)
  # dummy input to build the model
  dummy_input = tf.random.normal((batch_size, 10000, 3))
  model(dummy_input)  # Build the model with dummy input

  # loading dataset
  numbers = np.array([i for i in range(1, n_epochs+1)]) # 1 to n_epochs

  data1 = np.load('Chaos_Signals/data1_'+ '1' + '.npy')
  data2 = np.load('Chaos_Signals/data2_'+ '1' + '.npy')
  x_train = data1 + data2
  y_train = np.concatenate([data1, data2], 2)

  # MINE
  mymine = MINE_calculator()
  mymine.init_m()

  # preparation for analysis
  losses = []
  mis = []
  temp_list = []

  Q_cors = []
  Q_strs = []
  D_outs = [] 
  D_cors = []
  R2s = []

  node1 = {str(i+1) for i in range(N//2)}
  node2 = {str(i+1) for i in range(N//2, N)}

  w_recorder = WeightRecorder()
  w_recorder.record(model,0)

  # loading test dataset 
  data1_test = np.load('Chaos_Signals/data1_120.npy')
  data2_test = np.load('Chaos_Signals/data2_120.npy')
  x_test = data1_test + data2_test
  y_test = np.concatenate([data1_test, data2_test], 2)
  x_test_batch = x_test[:batch_size]
  y_test_batch = y_test[:batch_size]

  g1 = tf.random.get_global_generator()
  g2 = tf.random.get_global_generator()
  
  # initial MINE training
  # MINE training
  t_list = []
  for MINE_step1 in range(100):
    X_batch_m1, y_batch_m1 = random_batch(x_train, y_train, batch_size)
    middle_layer_t1 = model.hidden_layer(X_batch_m1)
    x_t1 = middle_layer_t1[:,:,:N//2] + g1.normal(shape=(batch_size,10000,N//2)) * s_mine # 0.05 is noise strength #type:ignore
    y_t1 = middle_layer_t1[:,:,N//2:] + g2.normal(shape=(batch_size,10000,N//2)) * s_mine # type:ignore
    mymine.train(x_t1,y_t1)
    mival_t, T = mymine(x_t1,y_t1)
    t_list.append(T)
    # print(mival_t)
    temp_list.append(mival_t)

  plt.plot(temp_list)
  plt.title('mi')
  plt.ylabel('MI')
  plt.xlabel('step')
  plt.show()  


  #  main training loop

  @tf.function
  def train_step(x_batch, y_batch):
  
    with tf.GradientTape() as tape_r:
        y_pred = model(x_batch)

        x = model.x[:,:,:N//2] + g1.normal(shape=(batch_size,10000,N//2)) * s_mine #type: ignore
        y = model.x[:,:,N//2:] + g2.normal(shape=(batch_size,10000,N//2)) * s_mine #type: ignore
        mival, T = mymine(x,y)

        # print(mival)
        main_loss = tf.add(tf.reduce_mean(loss_fn(y_batch[:,200:,:], y_pred[:,200:,:])),(lambda_I * mival[0]))
        loss = tf.add_n([main_loss] + model.losses)
    gradients_r = tape_r.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients_r, model.trainable_variables))
    return loss, y_pred, mival
  
  @tf.function
  def test_step(x_test_batch, y_test_batch):  
    y_pred = model(x_test_batch)

    x1 = model.x[:,:,:N//2] + g1.normal(shape=(batch_size,10000,N//2)) * s_mine #type: ignore
    x2 = model.x[:,:,N//2:] + g2.normal(shape=(batch_size,10000,N//2)) * s_mine #type: ignore
    mival, T = mymine(x1,x2)

    test_loss = tf.reduce_mean(loss_fn(y_test_batch[:,200:,:], y_pred[:,200:,:]))
    

    return test_loss, mival, y_pred



  #
  np.random.shuffle(numbers)

  for epoch in range(n_epochs):
      print("Epoch {}/{}".format(epoch+1, n_epochs))
      # loading dataset
      data1 = np.load('Chaos_Signals/data1_'+ str(1+ (numbers[epoch]-1)//100) + '.npy')
      data2 = np.load('Chaos_Signals/data2_'+ str(1+ (numbers[epoch]-1)//100) + '.npy')
      x_train = data1 + data2
      y_train = np.concatenate([data1, data2], 2)

      mi_mean.reset_states()
      mean_loss.reset_states()
      n_steps = data_size // batch_size


      for step in range( n_steps):

        # submodel (MINE) training
        for MINE_step2 in range(n_sm):
            X_batch_m, y_batch_m = random_batch(x_train, y_train, batch_size)
            middle_layer_t = model.hidden_layer(X_batch_m)
            x_t = middle_layer_t[:,:,:N//2] + g1.normal(shape=(batch_size,10000,N//2)) * s_mine # type: ignore
            y_t = middle_layer_t[:,:,N//2:] + g2.normal(shape=(batch_size,10000,N//2)) * s_mine #type:ignore
            mymine.train(x_t,y_t)
            mival, T = mymine(x_t,y_t)
        mi_mean(mival[0]) # add values
        # main model training
        
        # i = step % (len(x_train) // batch_size)  # Ensure we don't go out of bounds
        # start_idx = i * batch_size
        # end_idx = start_idx + batch_size
        # x_batch, y_batch = x_train[start_idx:end_idx], y_train[start_idx:end_idx]
        x_batch, y_batch = random_batch(x_train, y_train, batch_size)


        loss, y_pred, mi = train_step(x_batch, y_batch) # type:ignore

        for variable in model.variables:
            if variable.constraint is not None:
                variable.assign(variable.constraint(variable))

        mean_loss(loss) # metrics
        # losses.append(mean_loss.result())
        for metric in metrics:
            metric(y_batch, y_pred)
        print_status_bar((step+1) * batch_size, len(y_train), mean_loss, metrics)

      # using test data for evaluation
      test_loss, mival, y_pred_test = test_step(x_test_batch, y_test_batch) # type:ignore
      # R-squared scores for each output variable
      r2_test = r2_score(y_test_batch, y_pred_test)
      R2s.append(r2_test)
      # analysis
      Q_cor, Q_str, D_cor, D_out = calc_Q_and_D(model, x_test_batch, node1, node2, batch_size=batch_size)
      Q_cors.append(Q_cor)
      Q_strs.append(Q_str)
      D_outs.append(D_out)
      D_cors.append(D_cor)

      losses.append(mean_loss.result().numpy())
      mis.append(mi_mean.result().numpy())  
      for metric in [mean_loss] + metrics:
          metric.reset_states()
      mi_mean.reset_states()

      w_recorder.record(model,epoch+1)

      stats = {
        'Q_cors': Q_cors,
        'Q_strs': Q_strs,
        'D_outs': D_outs,
        'D_cors': D_cors,
        'losses': np.array(losses),
        'MI': np.array(mis),
        'R2s': np.array(R2s),
      }
      # save statistics
      np.savez(os.path.join(data_dir, 'statistics.npz'), **stats)
      #how to load:
      # stats = dict(np.load(os.path.join(data_dir,'statistics.npz')))

      if epoch % 10 == 0 or epoch == n_epochs - 1:
        print("\nEpoch {}:".format(epoch + 1))
        print("Loss: {:.4f}, MI: {:.4f}".format(mean_loss.result(), mi_mean.result()))
        print("Test Loss: {:.4f}, MI: {:.4f}".format(test_loss, mival[0]))
        print("R2:", r2_test)
        print("Q_cor:", Q_cor)
        print("Q_str:", Q_str)
        print("D_out:", D_out)
        print("D_cor:", D_cor)
        save_path = os.path.join(checkpoint_dir, 'ckpt-current')
        model.save_weights(save_path)
      print()

  model.summary()

  return model, mymine, stats, x_test, y_test


  # %%
  
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

  # hyperparameters
  validation_split_rate=0
  batch_size = 50
  data_size = 1000
  #data_size = 500
  n_epochs = 100
  #n_epochs = 10

  lambda_I = 0.005 # regularization parameter for MINE
  lambda_L2 = 0.001 # L2 regularization parameter for GRU
  lambda_L2_output = 0.0001 # L2 regularization parameter for output layer
  s_mine = 0.05 # noise strength for MINE training  


  # train_size = int(data_size * (1 - validation_split_rate))
  # n_steps = data_size // batch_size

  
  set_rand_seed(SEED0)

  model, mymine, stats, x_test, y_test = training_gru_mine(data_size=data_size, 
                                                  n_epochs=n_epochs, 
                                                  batch_size=batch_size,
                                                  lambda_I=lambda_I,
                                                  lambda_L2=lambda_L2,
                                                  lambda_L2_output=lambda_L2_output,
                                                  s_mine=s_mine)
  # analyze results
  analyze_results(model, mymine, stats,
                    x_test=x_test, y_test=y_test,
                    batch_size=batch_size,
                    N=N,
                    data_dir=data_dir
  )
# %%
