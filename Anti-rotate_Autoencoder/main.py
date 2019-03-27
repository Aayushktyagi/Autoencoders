'''
get dataset
get model
train models
'''
import numpy as np
#load from single class autoencoder
#from Autoencoder_class import Autoencoder
from Autoencoder_class_multilayer import Autoencoder
import matplotlib.pyplot as plt
from load_data import getData
import cv2

X_train,Y_train,X_test , Y_test = getData()
#show image
cv2.imshow("orignal_image ",X_train[0])
cv2.imshow("rotated image ",Y_train[0])
cv2.waitKey(0)
print("shape of orignal image:{},rotated image:{}".format(np.shape(X_train),np.shape(Y_train)))
#vectorize data

X_train = np.reshape(X_train,((len(X_train),784)))
Y_train = np.reshape(Y_train,((len(Y_train),784)))
X_test = np.reshape(X_test , ((len(X_test),784)))
Y_test = np.reshape(Y_test, ((len(Y_test),784)))
#print("shape of orignal image:{},rotated image:{}".format(np.shape(X_train),np.shape(Y_train)))

#train network
autoencoder = Autoencoder(np.shape(Y_train)[1],32)
autoencoder.train(Y_train,X_train,Y_test,X_test,256,10)
#uncomment for single class autoencoder
#encoded_image = autoencoder.getEncodedImage(Y_test)
decoder_image = autoencoder.getDecodedImage(X_test)


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
