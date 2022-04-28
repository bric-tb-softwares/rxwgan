

__all__ = ["Generator_v1", "Critic_v1"]

from turtle import width
from tensorflow.keras import layers
import tensorflow as tf

height=256
width =256

class Generator_v1( object ):

  def __init__(self, generator_path=None):

      self.latent_dim = 100
      self.height     = height
      self.width      = width
      self.leaky_relu_alpha = 0.3
    
      if generator_path:
        self.model = tf.keras.models.load_model(generator_path)
      else:
        self.compile()
      


  @tf.function
  def generate(self, nsamples):
    z = tf.random.normal( (nsamples, self.latent_dim) )
    return self.model( z )
  

  def compile(self):


      ip = layers.Input(shape=(self.latent_dim,))
      y = layers.Dense(units=32*32*32, input_shape=(self.latent_dim,))(ip)
      y = layers.Reshape(target_shape=(32,32, 32))(y)
      
      y = layers.Conv2DTranspose(16, (4,4), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
      y = layers.BatchNormalization()(y)
      y = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(y)
      y = layers.Dropout(rate=0.3)(y)

      y = layers.Conv2DTranspose(32, (4,4), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
      y = layers.BatchNormalization()(y)
      y = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(y)
      y = layers.Dropout(rate=0.3)(y)
      
      y = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
      y = layers.BatchNormalization()(y)
      y = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(y)
      y = layers.Dropout(rate=0.3)(y)

      out = layers.Conv2DTranspose(1, (4,4), strides=(1,1), padding='same', kernel_initializer='he_uniform', activation = 'tanh')(y)
      model = tf.keras.Model(ip, out)
      model.compile()
      model.summary()
      self.model = model





class Critic_v1( object ):

  def __init__(self, critic_path=None):

      self.height     = height
      self.width      = width
    
      if critic_path:
        self.model = tf.keras.models.load_model(critic_path)
      else:
        self.compile()

  def predict(self, samples, batch_size=1024):
    return self.model.predict(samples, verbose=1, batch_size=batch_size)

  def compile(self):


      ip = layers.Input(shape=( self.height,self.width,1))
      # TODO Add other normalization scheme as mentioned in the article
      # Input (None, 3^2*2^5 = 1 day = 288 samples, 1)
      y = layers.Conv2D(64, (5,5), strides=(2,2), padding='same', kernel_initializer='he_uniform', data_format='channels_last', input_shape=(self.height,self.width,1))(ip)
      #y = layers.BatchNormalization()(y)
      y = layers.Activation('relu')(y)
      y = layers.Dropout(rate=0.3, seed=1)(y)
      # Output (None, 3^2*2^3, 64)
      y = layers.Conv2D(32, (5,5), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
      #y = layers.BatchNormalization()(y)
      y = layers.Activation('relu')(y)
      y = layers.Dropout(rate=0.3, seed=1)(y)
      # Output (None, 3^2*2^3, 64)
      y = layers.Conv2D(16, (5,5), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
      #y = layers.BatchNormalization()(y)
      y = layers.Activation('relu')(y)
      y = layers.Dropout(rate=0.3, seed=1)(y)
      # Output (None, 3^2*2, 128)
      y = layers.Flatten()(y)
      # Output (None, 3*256)
      out = layers.Dense(1, activation='linear')(y)
      # Output (None, 1)
      model = tf.keras.Model(ip, out)
      model.compile()
      model.summary()
      self.model = model

