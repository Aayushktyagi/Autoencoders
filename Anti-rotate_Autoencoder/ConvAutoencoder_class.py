import numpy as np
from tensorflow.keras.layers import Input , Conv2D , MaxPooling2D , UpSampling2D
from tensorflow.keras.models import Model

class ConvAutoencoder(object):

    def __init__(self):
        '''
        Convolutional Autoencoder

        '''
        #Encoder
        input_layer = Input(shape=(28,28,1))
        encoding_conv_layer1 = Conv2D(16 , (3,3) , activation = 'relu' , padding = 'same')(input_layer)
        encoding_pooling_layer1 = MaxPooling2D((2,2) , padding = 'same')(encoding_conv_layer1)
        encoding_conv_layer2 = Conv2D(8,(3,3) , activation = 'relu' , padding = 'same')(encoding_pooling_layer1)
        encoding_pooling_layer2 = MaxPooling2D((2,2) , padding = 'same')(encoding_conv_layer2)
        encoding_conv_layer3 = Conv2D(8,(3,3) , activation = 'relu' ,padding = 'same')(encoding_pooling_layer2)
        latent_vector = MaxPooling2D((2,2) , padding = 'same')(encoding_conv_layer3)

        #Decoding
        decoding_conv_layer1 = Conv2D(8 , (3,3) ,activation = 'relu' , padding = 'same')(latent_vector)
        decoding_upsampling_layer1 = UpSampling2D((2,2))(decoding_conv_layer1)
        decoding_conv_layer2 = Conv2D(8 , (3,3) , activation = 'relu' , padding = 'same')(decoding_upsampling_layer1)
        decoding_upsampling_layer2 = UpSampling2D((2,2))(decoding_conv_layer2)
        decoding_conv_layer3 = Conv2D(16,(3,3) , activation ='relu')(decoding_upsampling_layer2)
        decoding_upsampling_layer3 = UpSampling2D((2,2))(decoding_conv_layer3)
        output_layer = Conv2D(1,(3,3) , activation = 'sigmoid',padding='same')(decoding_upsampling_layer3)


        self._model = Model(input_layer , output_layer)
        self._model.compile(optimizer = 'adam' , loss = 'binary_crossentropy')
        self._model.summary()

    def train(self, input_train ,output_train, input_test,output_test , batch_size , epochs):
        self._model.fit(input_train,
                        output_train ,
                        epochs = epochs,
                        batch_size = batch_size,
                        validation_data = (
                            input_test,
                            output_test))
    def getDecodedImage(self, encoded_image):
        decoded_image = self._model.predict(encoded_image)
        return decoded_image
