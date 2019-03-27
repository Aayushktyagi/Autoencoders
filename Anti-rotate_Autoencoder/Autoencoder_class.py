import numpy as np
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.models  import Model


class Autoencoder(object):

    def __init__(self,input_dim,latent_dim):
        '''
        Autoencoder class contains
        Input _layer > hidden layer > output layer 
        '''
        input_layer = Input(shape=(input_dim,))
        hidden_input = Input(shape=(latent_dim,))
        hidden_layer = Dense(latent_dim,activation='relu')(input_layer)
        output_layer = Dense(input_dim,activation = 'sigmoid')(hidden_layer)

        self.__autoencoder_model = Model(input_layer,output_layer)
        self.__encoder_model = Model(input_layer,hidden_layer)
        decoder_layer_out = self.__autoencoder_model.layers[-1]
        self.__decoder_model = Model(hidden_input,decoder_layer_out(hidden_input))

        self.__autoencoder_model.compile(optimizer = 'adam',loss = 'binary_crossentropy')

    def train(self,input_train,output_train,input_test,output_test,batch_size,epochs):
        self.__autoencoder_model.fit(input_train,
                                    output_train,
                                    epochs = epochs,
                                    batch_size = batch_size,
                                    validation_data = (input_test,
                                                    output_test))

    def getEncodedImage(self,image):
        encoded_image = self.__encoder_model.predict(image)
        return encoded_image

    def getDecodedImage(self,encoded_image):
        decoder_image = self.__decoder_model.predict(encoded_image)
        return decoder_image
