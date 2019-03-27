'''
load datatse
'''
import numpy as np
import cv2
from scipy.ndimage.interpolation import rotate
import tensorflow as tf

#config features
showImage = True

def rotate_image(image):
    X = np.zeros_like(image)
    for i in range(image.shape[0]):
        X[i,:,:] = rotate(image[i,:,:],np.random.randint(1,360),reshape=False)
        if i%1000 ==0:
            print("Number of images :{}".format(i))
    return X

def getData():
    #load datasets
    mnist = tf.keras.datasets.fashion_mnist
    (X_train,_),(X_test,_) = mnist.load_data()
    #normalize image
    X_train = X_train.astype('float32')/255.0
    X_test = X_test.astype('float32')/255.0
    X_train_rot_1 = rotate_image(X_train)
    X_train_rot_2 = rotate_image(X_train)
    X_train_orignal = np.concatenate([X_train,X_train],axis=0)
    X_train_rotate = np.concatenate([X_train_rot_1,X_train_rot_2],axis = 0)
    X_test_orignal = X_test
    X_test_rotate = rotate_image(X_test)
    return X_train_orignal , X_train_rotate , X_test_orignal , X_test_rotate
