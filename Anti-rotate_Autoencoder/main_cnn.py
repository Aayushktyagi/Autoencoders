'''
main file for convolutional Autoencoder
'''

from load_data import getData
import numpy as np
import cv2
from ConvAutoencoder_class import ConvAutoencoder
import matplotlib.pyplot as plt


#load dataset
X_train , Y_train , X_test , Y_test = getData()

#creating input tensor
X_train = np.reshape(X_train , (len(X_train) , 28,28,1))
Y_train = np.reshape(Y_train , (len(Y_train) , 28,28,1))
X_test = np.reshape(X_test , (len(X_test) , 28,28,1))
Y_test = np.reshape(Y_test , (len(Y_test) , 28,28,1))
print("Train data orignal:{},rotated:{}".format(np.shape(X_train),np.shape(Y_train)))

#train models
autoencoder = ConvAutoencoder()
autoencoder.train(Y_train ,X_train, Y_test,X_test , 256,10)
decoder_image = autoencoder.getDecodedImage(Y_test)

#visualization
plt.figure(figsize = (20,4))

for i in range(10):
    #orignal
    subplot = plt.subplot(2,10,i+1)
    plt.imshow(Y_train[i].reshape(28,28))
    plt.gray()
    subplot.get_xaxis().set_visible(False)
    subplot.get_yaxis().set_visible(False)
    #reconstructed image
    subplot = plt.subplot(2,10,i+11)
    plt.imshow(decoder_image[i].reshape(28,28))
    plt.gray()
    subplot.get_xaxis().set_visible(False)
    subplot.get_yaxis().set_visible(False)
plt.show()
