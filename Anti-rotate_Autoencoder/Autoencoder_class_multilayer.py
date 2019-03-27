import numpy as np
from tensorflow.keras.layers import Dense , Input
from tensorflow.keras.models import Model

class Autoencoder(object):

    def __init__(self,input_dim , latent_dim):

        '''
        Autoencoder architecture:
        input_layer > layer_1 > layer_2 > hidden_layer > layer_3 > layer_4 > output_layer

        '''
        input_img = Input(shape=(input_dim,))
        encoder = Dense(512,activation = 'relu')(input_img)
        encoder = Dense(128 , activation = 'relu')(encoder)

        latent = Dense(32 , activation = 'relu')(encoder)

        decoder = Dense(128 ,activation = 'relu')(latent)
        decoder = Dense(512 ,activation = 'relu')(decoder)
        output = Dense(784 , activation = 'sigmoid')(decoder)


        self.__autoencoder_model = Model(input_img , output)
        self.__autoencoder_model.compile(optimizer = 'adam' , loss = 'binary_crossentropy')

    def train(self,input_train , output_train , input_test , output_test , batch_size , epochs):
        self.__autoencoder_model.fit(input_train,
                                    output_train,
                                    epochs = epochs,
                                    batch_size = batch_size,
                                    validation_data = (input_test,
                                                    output_test))

    def getDecodedImage(self,image):
        decoded_image = self.__autoencoder_model.predict(image)
        return decoded_image  
