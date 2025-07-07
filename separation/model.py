#Model
import tensorflow as tf
from tensorflow.keras.layers import GRU, Dense
from tensorflow import keras


class MyModel(tf.keras.Model):
  def __init__(self,N=100, lambda_L2=0.001, lambda_L2_output=0.0001):
      super(MyModel, self).__init__()
      self.gru = GRU(N, 
                      return_sequences=True, 
                      input_shape=(None,3),
                      kernel_regularizer=keras.regularizers.l2(lambda_L2),
                      recurrent_regularizer=keras.regularizers.l2(lambda_L2))
      self.dense1 = Dense(6, kernel_regularizer=keras.regularizers.l2(lambda_L2_output))

  def call(self, inputs):
      self.x = self.gru(inputs)
      return self.dense1(self.x)

  def hidden_layer(self, input):
      return self.gru(input)
