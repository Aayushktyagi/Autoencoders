'''
Autoencoder implementation using tensorflow
'''


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#load mnist Datasets
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST",one_hot = True)

#hyper parameters:
num_input = 784
n_layer_1 = 256
n_layer_2 = 128
n_layer_3 = 64
n_epochs = 1
n_learning_rate = 0.01
batch_size = 50
display_step = 50

#input placeholder
X = tf.placeholder(tf.float32 , [None,num_input])

#weights
weights = {
    'encoder_h1':tf.Variable(tf.random_normal([num_input,n_layer_1])),
    'encoder_h2':tf.Variable(tf.random_normal([n_layer_1,n_layer_2])),
    'encoder_h3':tf.Variable(tf.random_normal([n_layer_2 , n_layer_3])),
    'decoder_h1':tf.Variable(tf.random_normal([n_layer_3,n_layer_2])),
    'decoder_h2':tf.Variable(tf.random_normal([n_layer_2,n_layer_1])),
    'decoder_h3':tf.Variable(tf.random_normal([n_layer_1,num_input]))
}
bias = {
    'encoder_b1':tf.Variable(tf.zeros([n_layer_1])),
    'encoder_b2':tf.Variable(tf.zeros([n_layer_2])),
    'encoder_b3':tf.Variable(tf.zeros([n_layer_3])),
    'decoder_b1':tf.Variable(tf.zeros([n_layer_2])),
    'decoder_b2':tf.Variable(tf.zeros([n_layer_1])),
    'decoder_b3':tf.Variable(tf.zeros([num_input]))
}
def encoder(x , weights, bias):
    layer_1 = tf.add(tf.matmul(x,weights['encoder_h1']),bias['encoder_b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1,weights['encoder_h2']),bias['encoder_b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2 , weights['encoder_h3']),bias['encoder_b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    return layer_3

def decoder(x , weights , bias):
    layer_1 = tf.add(tf.matmul(x , weights['decoder_h1']) , bias['decoder_b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1 , weights['decoder_h2']),bias['decoder_b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    layer_3 = tf.add(tf.matmul(layer_2,weights['decoder_h3']),bias['decoder_b3'])
    layer_3 = tf.nn.relu(layer_3)

    return layer_3

encoder_op = encoder(X , weights,bias)
decoder_op = decoder(encoder_op , weights,bias)

y_pred = decoder_op
y_true = X
#cost function
cost = tf.reduce_mean(tf.pow(tf.subtract(y_true,y_pred),2))
optimizer = tf.train.AdamOptimizer(n_learning_rate).minimize(cost)

#initialization
init = tf.global_variables_initializer()
#Save model
saver = tf.train.Saver()
#Staring training
with tf.Session() as sess:
    sess.run(init)
    for i  in range(n_epochs):
        epoch_loss = 0
        for _ in range(int(mnist.train.num_examples / batch_size)):
            batch_x , _ = mnist.train.next_batch(batch_size)
            #Run optimizatio and loss
            _ , l = sess.run([optimizer,cost],feed_dict={X:batch_x})
            epoch_loss +=l
        print("epoch :{} , loss:{}".format(i,epoch_loss))
        #Display batch_loss
        # if i % display_step == 0 or i ==0:
        #     print("Epoch :{} , loss :{}".format(i,l))
    save_path = saver.save(sess,"./Model/Autoencoder.ckpt")
    correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true,1)) # y' = y then correct
    accuracy = tf.reduce_mean(tf.cast(correct, 'float')) #cast to float correct
    print('Accuracy: ',accuracy.eval({X:mnist.test.images, X:mnist.test.images})) #eval using x
	#test data
    index = np.random.randint(mnist.test.images.shape[0], size=8) #random 8 number from test data to be predicted
    autoencoder = sess.run(y_pred, feed_dict={X: mnist.test.images[index]}) #predicted test data using model
    for i in range(8):
    	plt.imsave("./Results/Results"+str(i)+".png",np.reshape(autoencoder[i],(28,28)), cmap=plt.get_cmap('gray')) #save result


    #Starting testing
    # encode and decode images from test image set and visualization of reconstruction
    n = 4
    canvas_orignal = np.empty((28*n,28*n))
    canvas_recon = np.empty((28*n , 28*n))

    for i in range(n):
        #Get test dataset
        batch_x , _ = mnist.test.next_batch(n)
        g = sess.run(decoder_op , feed_dict = {X:batch_x})
        #Display  orignal
        for j in range(n):
            canvas_orignal[i*28:(i+1)*28,j*28:(j+1)*28] = batch_x[j].reshape([28,28])
            canvas_recon[i*28:(i+1)*28,j*28:(j+1)*28] = g[j].reshape([28,28])

    print("Orignal Images :")
    plt.figure(figsize = (n,n))
    plt.imshow(canvas_orignal,origin = "upper" , cmap = "gray")
    plt.show()
    plt.imsave("./Results/Results_canvas_orignal"+".png",canvas_orignal)
    #plt.savefig("./Results/Results_canvas_orignal.png")

    print("Reconstructed Images :")
    plt.figure(figsize = (n,n))
    plt.imshow(canvas_recon,origin = "upper" , cmap = 'gray')
    plt.show()
    plt.imsave("./Results/Results_canvas_recon"+".png",canvas_recon)
    # plt.savefig("./Results/Results_canvas_recon.png")
