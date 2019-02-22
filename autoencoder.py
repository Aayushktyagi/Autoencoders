'''
Autoencoder implementation using tensorflow
'''


import tensorflow as tf
import numpy as np


#load mnist Datasets
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST",one_hot = True)

#hyper parameters:
num_input = 784
n_layer_1 = 256
n_layer_2 = 128
n_epochs = 10000
n_learning_rate = 0.001
batch_size = 256
display_step = 50

#input placeholder
X = tf.placeholder(tf.float32 , [None,num_input])

#weights
weights = {
    'encoder_h1':tf.Variable(tf.random_normal([num_input,n_layer_1])),
    'encoder_h2':tf.Variable(tf.random_normal([n_layer_1,n_layer_2])),
    'decoder_h1':tf.Variable(tf.random_normal([n_layer_2,n_layer_1])),
    'decoder_h2':tf.Variable(tf.random_normal([n_layer_1,num_input]))
}
bias = {
    'encoder_b1':tf.Variable(tf.random_normal([n_layer_1])),
    'encoder_b2':tf.Variable(tf.random_normal([n_layer_2])),
    'decoder_b1':tf.Variable(tf.random_normal([n_layer_1])),
    'decoder_b2':tf.Variable(tf.random_normal([num_input]))
}
def encoder(x , weights, bias):
    layer_1 = tf.add(tf.matmul(x,weights['encoder_h1']),bias['encoder_b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1,weights['encoder_h2']),bias['encoder_b2'])
    layer_2 = tf.nn.relu(layer_2)

    return layer_2

def decoder(x , weights , bias):
    layer_1 = tf.add(tf.matmul(x , weights['decoder_h1']) , bias['decoder_b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1 , weights['decoder_h2']),bias['decoder_b2'])
    layer_2 = tf.nn.relu(layer_2)

    return layer_2

encoder_op = encoder(X , weights,bias)
decoder_op = decoder(encoder_op , weights,bias)

y_pred = decoder_op
y_true = X
#cost function
cost = tf.reduce_mean(tf.pow(y_pred - y_true,2))
optimizer = tf.train.AdamOptimizer(n_learning_rate).minimize(cost)

#initialization
init = tf.global_variables_initializer()

#Staring training
with tf.Session() as sess:
    sess.run(init)
    for i  in range(0,n_epochs):
        batch_x , _ =mnist.train.next_batch(batch_size)
        #Run optimizatio and loss
        _ , l = sess.run([optimizer,cost],feed_dict={X:batch_x})
        #Display batch_loss
        if i % display_step == 0 or i ==0:
            print("Epoch :{} , loss :{}".format(i,l))