
__all__ = ['wgan']

import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def declare_property( cls, kw, name, value , private=False):
  atribute = ('__' + name ) if private else name
  if name in kw.keys():
    setattr(cls,atribute, kw[name])
  else:
    setattr(cls,atribute, value)




class wgan(object):

  def __init__(self, generator, discriminator, **kw ):

    declare_property(self, kw, 'height'              , 128                       )
    declare_property(self, kw, 'width'               , 128                       )
    declare_property(self, kw, 'max_epochs'          , 1000                      )
    declare_property(self, kw, 'batch_size'          , 3                         ) 
    declare_property(self, kw, 'n_discr'             , 0                         )
    declare_property(self, kw, 'save_interval'       , 9                         ) 
    declare_property(self, kw, 'use_gradient_penalty', True                      )
    declare_property(self, kw, 'verbose'             , True                      )
    declare_property(self, kw, 'leaky_relu_alpha'    , 0.3                       )
    declare_property(self, kw, 'grad_weight'         , 10.0                      )
    declare_property(self, kw, 'latent_dim'          , 100                       )
    declare_property(self, kw, 'gen_optimizer'       , tf.optimizers.Adam(lr=1e-4, beta_1=0.5, decay=1e-4 )     )
    declare_property(self, kw, 'discr_optimizer'     , tf.optimizers.Adam(lr=1e-4, beta_1=0.5, decay=1e-4 )     )


    # Initialize discriminator and generator networks
    self.discriminator = discriminator
    self.generator     = generator


  #
  # Train models
  #
  def train(self, train_generator ):


    gpus = tf.config.experimental.list_physical_devices('GPU')
    n_gpus = len(gpus)

    print('This machine has %i GPUs.' % n_gpus)

 
 
    # containers for losses
    losses = {'discr': [], 'generator': [], 'regularizer': []}
    discr_acc = []

    updates = 0
    batches = 0


    for epoch in range(self.max_epochs):

      for data_batch, target_batch in train_generator:

        if self.n_discr and ( (updates % self.n_discr)==0 ):
          # Update only discriminator
          discr_loss, reg_loss, gen_loss = self.train_discr(data_batch) + (np.nan,)

        if not(self.n_discr) or not( (updates % self.n_discr)==0 ):
          # Update discriminator and generator
          discr_loss, gen_loss, reg_loss = self.train_discr_and_gen(data_batch)


        losses['discr'].append(discr_loss)
        losses['generator'].append(gen_loss)
        losses['regularizer'].append(reg_loss)

        updates += 1

        if self.verbose and (updates % 25)==0 :
          perc = np.around(100*epoch/self.max_epochs, decimals=1)
          print('Epoch: %i. Updates %i. Training %1.1f%% complete. discr_loss: %.3f. Gen_loss: %.3f. Regularizer: %.3f'
               % (epoch, updates, perc, discr_loss, gen_loss, reg_loss ))


    return losses



  #
  # x = real sample, x_hat = generated sample
  #
  @tf.function
  def gradient_penalty(self, x, x_hat): 

    # 0.0 <= epsilon <= 1.0
    epsilon = tf.random.uniform((self.batch_size, self.height, self.width, 1), 0.0, 1.0) 
    # Google Search - u_hat is a randomly weighted average between a real and generated sample
    u_hat = epsilon * x + (1 - epsilon) * x_hat 
    with tf.GradientTape() as penalty_tape:
      penalty_tape.watch(u_hat)
      func = self.discriminator(u_hat)
    grads = penalty_tape.gradient(func, u_hat) #func gradient at u_hat
    norm_grads = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    # regularizer - to avoid overfitting
    regularizer = tf.math.square( tf.reduce_mean((norm_grads - 1) ) ) 
    return regularizer


  #
  # Calculate discriminator output
  #
  @tf.function
  def calculate_discr_output( self, samples, fake_samples ):
    # calculate discr outputs
    real_output = self.discriminator(samples)
    fake_output = self.discriminator(fake_samples)
    return real_output, fake_output


  #
  # Diff between discr output on real instances(real data) and fake instances(generator data)
  #
  @tf.function
  def wasserstein_loss(self, y_true, y_pred): 
    return tf.reduce_mean(y_true) - tf.reduce_mean(y_pred)

  #
  # Calculate the discriminator loss
  #
  @tf.function
  def calculate_discr_loss( self, samples, fake_samples, real_output, fake_output ):
    grad_regularizer_loss = tf.multiply(tf.constant(self.grad_weight), self.gradient_penalty(samples, fake_samples)) if self.use_gradient_penalty else 0
    discr_loss = tf.add( self.wasserstein_loss(real_output, fake_output), grad_regularizer_loss )
    return discr_loss, grad_regularizer_loss

  #
  # Calculate the generator loss
  #
  def calculate_gen_loss( self, fake_samples, fake_output ):
    gen_loss = tf.reduce_mean(fake_output)
    return gen_loss


  def discr_update( self, discr_tape, discr_loss ): #Update discr
    discr_grads = discr_tape.gradient(discr_loss, self.discriminator.trainable_variables)
    self.discr_optimizer.apply_gradients(zip(discr_grads, self.discriminator.trainable_variables))


  def gen_update( self, gen_tape, gen_loss): #Update generator
    gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
    self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))


  #
  # discr = Discriminator; Tries to distinguish real data from the data created by the generator
  #
  @tf.function
  def train_discr(self, samples): 
    
    with tf.GradientTape() as discr_tape:
      fake_samples = self.generate( self.batch_size )
      # real_output => output from real samples;fake_output => output from fake samples
      real_output, fake_output = self.calculate_discr_output( samples, fake_samples ) 
      # discr loss => wasserstein loss between real output and fake output + regularizer
      discr_loss, grad_regularizer_loss = self.calculate_discr_loss( samples, fake_samples, real_output, fake_output) 
    
    # discr_tape
    # Backpropagation(negative feedback??) to improve weights of the discr?
    self.discr_update( discr_tape, discr_loss ) 
    return discr_loss, grad_regularizer_loss


  #
  #
  #
  @tf.function
  def train_discr_and_gen(self, samples):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as discr_tape:
      fake_samples = self.generate( self.batch_size )
      real_output, fake_output = self.calculate_discr_output( samples, fake_samples )
      discr_loss, discr_regularizer = self.calculate_discr_loss( samples, fake_samples, real_output, fake_output)
      # gen_loss => Variable to improve the generator to try to make discr classify its fake samples as real;
      gen_loss = self.calculate_gen_loss( fake_samples, fake_output ) 
    
    # gen_tape, discr_tape
    self.discr_update( discr_tape, discr_loss )
    self.gen_update( gen_tape, gen_loss )
    return discr_loss, gen_loss, discr_regularizer



  @tf.function
  def generate(self, nsamples):
    z = tf.random.normal( (nsamples, self.latent_dim) )
    return self.generator( z )
  


