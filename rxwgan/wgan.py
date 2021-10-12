
__all__ = ['wgan_optimizer']

import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from rxwgan.utils import declare_property
import matplotlib.pyplot as plt
import os
import json


import atlas_mpl_style as ampl
ampl.use_atlas_style()



class wgan_optimizer(object):

  def __init__(self, model, **kw ):

    declare_property(self, kw, 'max_epochs'          , 1000                      )
    declare_property(self, kw, 'n_discr'             , 0                         )
    declare_property(self, kw, 'save_interval'       , 9                         ) 
    declare_property(self, kw, 'use_gradient_penalty', True                      )
    declare_property(self, kw, 'grad_weight'         , 10.0                      )
    declare_property(self, kw, 'gen_optimizer'       , tf.optimizers.Adam(lr=1e-4, beta_1=0.5, decay=1e-4 )     )
    declare_property(self, kw, 'discr_optimizer'     , tf.optimizers.Adam(lr=1e-4, beta_1=0.5, decay=1e-4 )     )
    declare_property(self, kw, 'disp_for_each'       , 0                         )
    declare_property(self, kw, 'save_for_each'       , 0                         )
    declare_property(self, kw, 'output_dir'          , 'output'                  )
    declare_property(self, kw, 'notebook'            , True                      )

    # Initialize discriminator and generator networks
    self.discriminator = model.discriminator
    self.generator     = model.generator
    self.latent_dim    = model.latent_dim
    self.height        = model.height
    self.width         = model.width
    self.model         = model
    gpus = tf.config.experimental.list_physical_devices('GPU')
    n_gpus = len(gpus)
    print('This machine has %i GPUs.' % n_gpus)


 

  #
  # Train models
  #
  def train(self, train_generator ):


    output = os.getcwd()+'/'+self.output_dir
    if not os.path.exists(output): os.makedirs(output)


    self.history = {
                'discr_loss' : [],
                'gen_loss'   : [],
                'reg'        : [],
    }



    for epoch in range(self.max_epochs):

      batches = 0
      vec_discr_loss = []
      vec_gen_loss = []
      vec_reg_loss = []

      for data_batch, _ in tqdm( train_generator , desc= 'training: ', ncols=60): 

        if self.n_discr and not ( (batches % self.n_discr)==0 ):
          # Update only discriminator
          discr_loss, reg_loss, gen_loss = self.train_discr(data_batch) + (np.nan,)
        else:
          # Update discriminator and generator
          discr_loss, gen_loss, reg_loss = self.train_discr_and_gen(data_batch)
        
        batches += 1

        # accumulate
        vec_discr_loss.append(discr_loss)
        vec_gen_loss.append(gen_loss)
        vec_reg_loss.append(reg_loss)

        # save last values
        if batches > len(train_generator):
          break
        
        
        # end of batch

      # Update the history
      self.history['discr_loss'].append( np.mean(vec_discr_loss) ) # wasserstein loss
      self.history['gen_loss'].append( np.mean(vec_gen_loss) )
      self.history['reg'].append( np.mean(vec_reg_loss) )
      
      perc = np.around(100*epoch/self.max_epochs, decimals=1)
      print('Epoch: %i. Training %1.1f%% complete. discr_loss: %.3f. Gen_loss: %.3f. Regularizer: %.3f'
               % (epoch, perc, self.history['discr_loss'][-1], self.history['gen_loss'][-1], 
                  self.history['reg'][-1] ))


      if self.disp_for_each and ( (epoch % self.disp_for_each)==0 ):
        self.display(epoch, output)

      if self.save_for_each and ( (epoch % self.save_for_each)==0 ):
        self.model.save(output)

    # just to be sure!
    self.model.save(output)
    return self.history



  #
  # x = real sample, x_hat = generated sample
  #
  @tf.function
  def gradient_penalty(self, x, x_hat): 
    batch_size = x.shape[0]
    # 0.0 <= epsilon <= 1.0
    epsilon = tf.random.uniform((batch_size, self.height, self.width, 1), 0.0, 1.0) 
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
  def calculate_discr_output( self, real_samples, fake_samples ):
    # calculate discr outputs
    real_output = self.discriminator(real_samples)
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
  def calculate_discr_loss( self, real_samples, fake_samples, real_output, fake_output ):
    grad_regularizer_loss = tf.multiply(tf.constant(self.grad_weight), self.gradient_penalty(real_samples, fake_samples)) if self.use_gradient_penalty else 0
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
  def train_discr(self, real_samples): 
    
    with tf.GradientTape() as discr_tape:
      batch_size = real_samples.shape[0]
      fake_samples = self.generate( batch_size )
      # real_output => output from real samples;fake_output => output from fake samples
      real_output, fake_output = self.calculate_discr_output( real_samples, fake_samples ) 
      # discr loss => wasserstein loss between real output and fake output + regularizer
      discr_loss, grad_regularizer_loss = self.calculate_discr_loss( real_samples, fake_samples, real_output, fake_output) 
    
    # discr_tape
    # Backpropagation(negative feedback??) to improve weights of the discr?
    self.discr_update( discr_tape, discr_loss ) 
    return discr_loss, grad_regularizer_loss


  #
  #
  #
  @tf.function
  def train_discr_and_gen(self, real_samples):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as discr_tape:
      batch_size = real_samples.shape[0]
      fake_samples = self.generate( batch_size )
      real_output, fake_output = self.calculate_discr_output( real_samples, fake_samples )
      discr_loss, discr_regularizer = self.calculate_discr_loss( real_samples, fake_samples, real_output, fake_output)
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
  


  def display(self, epoch, output):

    # disp plot
    fake_samples = self.generate(25)
    fig = plt.figure(figsize=(10, 10))
    for i in range(25):
       plt.subplot(5,5,1+i)
       plt.axis('off')
       plt.imshow(fake_samples[i],cmap='gray')
    if self.notebook:
      plt.show()
    fig.savefig(output + '/fake_samples_epoch_%d.pdf'%epoch,)

    if epoch > 0: # no make sense to save since the begginer
      self.plot_evolution(self.history, output+'/evolution_%d.pdf'%epoch)

  def plot_evolution( self, history, output ):
      fig = plt.figure(figsize=(10, 5))
      epochs = list(range(len(history['discr_loss'])))
      xmin = epochs[0] - 20
      xmax = epochs[-1] + 20
      plt.plot( epochs, history['discr_loss'], label='discr_loss', color='b')
      plt.plot( epochs, history['gen_loss'], label='gen_loss', color='r')
      plt.xlabel('Epochs',fontsize=18,loc='right')
      plt.ylabel('Loss',fontsize=18,loc='top')
      ax = plt.gca()
      ax.set_xlim(xmin,xmax)
      plt.legend()
      if self.notebook:
        plt.show()
      fig.savefig(output)