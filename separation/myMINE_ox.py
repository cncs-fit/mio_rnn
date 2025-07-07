#%%
from tensorflow import keras
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt


# T を計算するnet
class MINE_T(layers.Layer):
    def __init__(self, n_hidden, **kwargs):
        super(MINE_T, self).__init__(**kwargs)
        self.dense_x = layers.Dense(n_hidden)
        self.dense_y = layers.Dense(n_hidden) 
        self.relu = layers.ReLU(negative_slope=0.25)
        self.dense = layers.Dense(n_hidden)
        self.relu2 = layers.ReLU(negative_slope=0.25)
        self.dense_out = layers.Dense(1, use_bias=False)

    def call(self, x_in, y_in):
        lx = self.dense_x(x_in)
        ly = self.dense_y(y_in)      
        outputs = self.relu(lx+ly)
        outputs1 = self.dense(outputs)
        outputs2 = self.relu2(outputs1)
        return self.dense_out(outputs2)

# T をラップして MI推定を出す．
class MINE(tf.keras.Model):
    def __init__(self, t_net=None, n_hidden=None, **kwargs):
        super(MINE, self).__init__(**kwargs)
        if t_net is not None:
            self.T=t_net
        else:
            self.T=MINE_T(n_hidden=n_hidden)      

    def call(self, x_in, y_in):
        y_shuffle = tf.gather(y_in, tf.random.shuffle(tf.range(tf.shape(y_in)[0])))
        T_xy = self.T(x_in,y_in) # true distribution of XY
        T_x_y =self.T(x_in,y_shuffle) # pseudo-independent distribution of X and Y made by shuffling
        return  tf.reduce_mean(T_xy, axis=0)  -  tf.reduce_mean(T_x_y) - tf.math.log(tf.reduce_mean(tf.math.exp(T_x_y - tf.reduce_mean(T_x_y)), axis=0)) ,  tf.reduce_mean(T_x_y)# estimated mi 
    


class MINE_calculator():
    H=100
    data_size = 500000
    select_neuron = 50

    def __init__(self):
        self.mine = MINE(n_hidden=MINE_calculator.H)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, clipvalue=1.0)
        self.mi_estimate = tf.keras.metrics.Mean(name='mi_estimate')

    def __call__(self, x_in, y_in):
            self.x = tf.reshape(x_in, shape=(MINE_calculator.data_size,MINE_calculator.select_neuron))
            self.y = tf.reshape(y_in, shape=(MINE_calculator.data_size,MINE_calculator.select_neuron))
            return  self.mine(self.x, self.y) 

    def gen_x(self):
        return tf.reshape(self.x, shape=(MINE_calculator.data_size,MINE_calculator.select_neuron))

    def gen_y(self):
        return tf.reshape(self.y, shape=(MINE_calculator.data_size,MINE_calculator.select_neuron))
    
    @tf.function
    def calculator(self):
        self.x_t = self.gen_x()
        self.y_t = self.gen_y()    
        self.mine(self.x_t,self.y_t)

    @tf.function
    def train_step(self,x,y):
        with tf.GradientTape() as self.tape:
            self.mi, _ = self.mine(x,y)
            self.mi_objective =  -self.mi
        self.gradients = self.tape.gradient(self.mi_objective, self.mine.trainable_variables)
        self.optimizer.apply_gradients(zip(self.gradients, self.mine.trainable_variables))
        self.mi_estimate(self.mi)

    @tf.function
    def test_step(self, x,y):
        self.mi, _ = self.mine(x,y)
        self.mi_estimate_test(self.mi)

    def init_m(self):
        self.mi_estimate = tf.keras.metrics.Mean(name='mi_estimate')
        self.mi_estimate_test = tf.keras.metrics.Mean(name='mi_estimate_test')
        self.mis=[]

    @tf.function     
    def train(self, x, y):
        self.x = x
        self.y = y        
        x_sample = self.gen_x()
        y_sample = self.gen_y()
        self.train_step(x_sample, y_sample)
        self.mis.append(self.mi_estimate.result())
        self.mi_estimate.reset_states()

    @tf.function
    def mi_estimation(self, n_epochs):
        self.mi_estimate_test.reset_states()
        for epoch in range(n_epochs):
            ## generate the data
            x_sample = self.gen_x()
            y_sample = self.gen_y()
            self.test_step(x_sample, y_sample)
        mi_val = self.mi_estimate_test.result()    
        self.mi_estimate_test.reset_states()
        return mi_val


# mid_out = np.load('mid_out10_t.npy')
# x = tf.constant(mid_out[:20,:50,:].astype(np.float32))
# print(x.shape)
# print(type(x))
# y = tf.constant(mid_out[:20,100:,:].astype(np.float32))
#%%
# mymine = MINE_calculator()

#%%
# mymine.train(x,y)
# %%

# print(mymine(x,y))
# mival = mymine.mi_estimation(100)
# print(mival)

# %%

# %%
