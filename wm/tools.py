import tensorflow as tf

# splitting long sequences into multiple short subsequences
def split_sequence(seq, split_length):
  # seq is expected to have (batch, timestep, dim) shape.
  # returned list have (len_stack) elements, which have (batch, split_length, dim) shape.
  len_stack = seq.shape[1] // split_length
  split_sizes = [split_length]*len_stack

  if seq.shape[1] % split_length != 0:
    # last subsequence which has shorter length
    split_sizes.append(seq.shape[1] % split_length)
  seq_stack = tf.split(seq, split_sizes, axis=1)
  return seq_stack


def get_input(task, gp):
  '''get input and target signal from task object and split it if it is longer than gp.split_length
  returns:
    inputs_stack: list of input signal
    target_stack: list of target signal
    inputs_transient: input signal for transient period. numpy array (bs, len_transient, dim)
    target_transient: target signal for transient period. numpy array (bs, len_transient, dim)
  '''
  task.gen_signals()
  inputs_transient, target_transient = task.signals_transient()
  inputs, target = task.signals_learning()
  inputs_stack = split_sequence(inputs, gp.split_length)
  target_stack = split_sequence(target, gp.split_length)

  return inputs_stack, target_stack, inputs_transient, target_transient

import os
import shutil

def clean_saved_directories():
  ''' clean contents of checkpoints, data, figures directories  
  '''
  # clean checkpoints directory
  if os.path.exists('checkpoints'):
    shutil.rmtree('checkpoints')
    os.mkdir('checkpoints')
    
  # clean data directory
  if os.path.exists('data'):
    shutil.rmtree('data')
    os.makedirs('data')
  
  # clean figures directory
  if os.path.exists('figures'):
    shutil.rmtree('figures')
    os.makedirs('figures')

  # clean multiple_trials directory
  # if os.path.exists('multiple_trials'):
  #   shutil.rmtree('multiple_trials')
  #   os.makedirs('multiple_trials')  

  
if __name__ == '__main__':
  clean_saved_directories()