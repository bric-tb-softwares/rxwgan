
__all__ = ['Model_v1']


from tensorflow.keras import layers
import tensorflow as tf



class Model_v1( object ):

  def __init__(self, trainable = True):

      self.latent_dim = 100
      self.height     = 128
      self.width      = 128
      self.leaky_relu_alpha = 0.3

      if trainable:
        self.compile_generator()
        self.generator.summary()
        self.compile_discriminator()
        self.discriminator.summary()

      self.trainable = trainable

  #
  # gen.save( /path/to/model/)
  #
  def save(self, output_path):
    self.generator.save(output_path + '/generator.h5')
    self.discriminator.save(output_path + '/discriminator.h5')
    

  def load(self, input_path ):
    self.generator = tf.keras.models.load_model(input_path + '/generator.h5')
    self.discriminator = tf.keras.models.load_model(input_path + '/discriminator.h5')



  @tf.function
  def generate(self, nsamples):
    z = tf.random.normal( (nsamples, self.latent_dim) )
    return self.generator( z )
  

  def compile_generator( self ):

      ip = layers.Input(shape=(self.latent_dim,))
      # Input (None, latent space (100?) )
      y = layers.Dense(units=16*16*32, input_shape=(self.latent_dim,))(ip)
      # Output (None, 64*3^2 )
      y = layers.Reshape(target_shape=(16,16, 32))(y)
      #y = layers.BatchNormalization()(y)
      #y = layers.LeakyReLU(alpha=leaky_relu_alpha)(y)
      #y = layers.UpSampling1D()(y)
      # Output (None, 3^2*2, 64)
      y = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
      y = layers.BatchNormalization()(y)
      y = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(y)
      y = layers.Dropout(rate=0.3)(y)
      #y = layers.UpSampling1D(size=2*2)(y)
      # Output (None, 3^2*2^3, 128)
      y = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
      y = layers.BatchNormalization()(y)
      y = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(y)
      y = layers.Dropout(rate=0.3)(y)
      #y = layers.UpSampling1D(size=2*2)(y)
      # Output (None, 3^2*2^5, 256)
      y = layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
      y = layers.BatchNormalization()(y)
      y = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(y)
      y = layers.Dropout(rate=0.3)(y)
      # Output (None, 3^2*2^5, 64)
      out = layers.Conv2DTranspose(1, (4,4), strides=(1,1), padding='same', kernel_initializer='he_uniform', activation = 'tanh')(y)
      # Output (None, 3^2*2^5, 1)
      model = tf.keras.Model(ip, out)
      model.compile()
      self.generator = model




  def compile_discriminator(self):

      ip = layers.Input(shape=( self.height,self.width,1))
      # TODO Add other normalization scheme as mentioned in the article
      # Input (None, 3^2*2^5 = 1 day = 288 samples, 1)
      y = layers.Conv2D(256, (5,5), strides=(2,2), padding='same', kernel_initializer='he_uniform', data_format='channels_last', input_shape=(self.height,self.width,1))(ip)
      #y = layers.BatchNormalization()(y)
      y = layers.Activation('relu')(y)
      y = layers.Dropout(rate=0.3, seed=1)(y)
      # Output (None, 3^2*2^3, 64)
      y = layers.Conv2D(128, (5,5), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
      #y = layers.BatchNormalization()(y)
      y = layers.Activation('relu')(y)
      y = layers.Dropout(rate=0.3, seed=1)(y)
      # Output (None, 3^2*2^3, 64)
      y = layers.Conv2D(64, (5,5), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
      #y = layers.BatchNormalization()(y)
      y = layers.Activation('relu')(y)
      y = layers.Dropout(rate=0.3, seed=1)(y)
      # Output (None, 3^2*2, 128)
      y = layers.Flatten()(y)
      # Output (None, 3*256)
      #out = layers.Dense(nb_class, activation='sigmoid')(y)
      out = layers.Dense(1, activation='linear')(y)
      # Output (None, 1)
      model = tf.keras.Model(ip, out)
      model.compile()
      #y = layers.GlobalAveragePooling1D()(y)
      self.discriminator = model
