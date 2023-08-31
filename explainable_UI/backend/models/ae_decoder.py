# -*- coding: utf-8 -*-
"""AE_decoder.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16ydZQjoADzwn9bS2Yeu5nAnhPFWjX49g
"""

feature_extractor = tf.keras.Model(
   inputs= sota_classifier.inputs,
   outputs= sota_classifier.get_layer(name='fully_connected_3').output,
)

class Autoencoder(Model):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.decoder = tf.keras.Sequential([
    
                                        
      Dense(units = 256, name = 'fully_connected_1_decoder', use_bias=False),
      Activation(LeakyReLU(alpha= 0.2)),
      BatchNormalization(name = 'batchnorm_3_decoder'),

      Dense(units = 4*4*64, name = 'fully_connected_decoder', use_bias=False), 
      Reshape((4, 4, 64)),

      Conv2DTranspose(filters = 128, kernel_size = 4, strides = 1 ,activation= LeakyReLU(alpha= 0.2), name = 'convolution_4_decoder', use_bias=False),  #'relu'
      BatchNormalization(name = 'batchnorm_3_decoder'),
      
      
      Conv2DTranspose(filters = 64, kernel_size = (3,3), strides = 2 ,padding = 'same', activation= LeakyReLU(alpha= 0.2), name = 'convolution_3_decoder', use_bias=False),  #'relu'
      BatchNormalization(name = 'batchnorm_2_decoder'),

      Conv2DTranspose(filters = 32, kernel_size = (3,3), strides = 2, padding = 'same', activation= LeakyReLU(alpha= 0.2), name = 'convolution_2_decoder', use_bias=False),  #'relu'
      BatchNormalization(name = 'batchnorm_1_decoder'),
     
      Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same', name = 'convolution_1_decoder'),
    ])

  def call(self, x):
    encoded = feature_extractor(x)
    decoded = self.decoder(encoded)
    return decoded
