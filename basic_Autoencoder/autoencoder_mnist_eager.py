'''
autoencoder sequential
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



np.random.seed(1)
tf.random.set_seed(1)
batch_size = 128
epochs = 100
learning_rate = 1e-3
momentum = 9e-1
intermediate_dim = 64
original_dim = 784
hidden_layer_2_dim = 512
hidden_layer_3_dim = 150

#load mnist dataset
(training_features,_),(test_features,_) = tf.keras.datasets.mnist.load_data()
training_features = training_features / np.max(training_features)
training_features = training_features.reshape(training_features.shape[0],training_features.shape[1]*training_features.shape[2]).astype(np.float32)
training_dataset = tf.data.Dataset.from_tensor_slices(training_features).batch(batch_size)

#model
class Encoder(tf.keras.layers.Layer):
    def __init__(self,input_dim, intermediate_dim):
        super(Encoder,self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(units=input_dim,activation = tf.nn.relu)
        self.hidden_layer_2 = tf.keras.layers.Dense(units = hidden_layer_2_dim,activation = tf.nn.relu)
        self.hidden_layer_3 = tf.keras.layers.Dense(units = hidden_layer_3_dim,activation = tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units = intermediate_dim,activation = tf.nn.relu)
    def call(self, input_features):
        activation1 = self.hidden_layer(input_features)
        activation2 = self.hidden_layer_2(activation1)
        activation3 = self.hidden_layer_3(activation2)
        return self.output_layer(activation3)

class Decoder(tf.keras.layers.Layer):
    def __init__(self,intermediate_dim,output_dim):
        super(Decoder,self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(units = intermediate_dim,activation = tf.nn.relu)
        self.hidden_layer_2 = tf.keras.layers.Dense(units =hidden_layer_2_dim , activation = tf.nn.relu)
        self.hidden_layer_3 = tf.keras.layers.Dense(units = hidden_layer_3_dim,activation = tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units = output_dim,activation=tf.nn.relu )
    def call(self,code):
        activation1 = self.hidden_layer(code)
        activation2 = self.hidden_layer_2(activation1)
        activation3 = self.hidden_layer_3(activation2)
        return self.output_layer(activation3)

class Autoencoder(tf.keras.Model):
    def __init__(self,input_dim,intermediate_dim,output_dim):
        super(Autoencoder,self).__init__()
        self.encoder = Encoder(input_dim = input_dim,intermediate_dim = intermediate_dim)
        self.decoder = Decoder(intermediate_dim = intermediate_dim , output_dim = output_dim)
    def call(self,input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed

#define loss function
def loss(model,orignal):
    reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(orignal),orignal)))
    return reconstruction_error

#gradient
def train(loss,model,opt,orignal):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model,orignal),model.trainable_variables)
    gradient_variables = zip(gradients,model.trainable_variables)
    opt.apply_gradients(gradient_variables)

#training and visualization
autoencoder = Autoencoder(input_dim = 784, intermediate_dim = 64 , output_dim = 784)
opt = tf.optimizers.SGD(learning_rate = learning_rate , momentum = momentum)


#summery and training
writer = tf.summary.create_file_writer('tmp')

with writer.as_default():
    with tf.summary.record_if(True):
        for epoch in range(epochs):
            print("Epochs :{}".format(epoch))
            for step , batch_features in enumerate(training_dataset):
                train(loss,autoencoder,opt,batch_features)
                loss_values = loss(autoencoder,batch_features)
                print("loss:{}".format(loss_values))
                orignal = tf.reshape(batch_features,(batch_features.shape[0],28,28,1))
                reconstructed = tf.reshape(autoencoder(tf.constant(batch_features)),(batch_features.shape[0],28,28,1))
                tf.summary.scalar('loss',loss_values,step = step)
                tf.summary.image('orignal',orignal,max_outputs = 10,step =step)
                tf.summary.image('reconstructed',reconstructed,max_outputs=10,step = step)
