
__all__ = ['wgangp_optimizer']

import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from rxwgan.utils import declare_property
from rxwgan.plots import plot_evolution
from rxwgan.stats import calc_kl , calc_js, est_pdf
import matplotlib.pyplot as plt
import os
import json

import atlas_mpl_style as ampl
ampl.use_atlas_style()




class wgangp_optimizer(object):

  def __init__(self, critic, generator, **kw ):

    declare_property(self, kw, 'max_epochs'          , 1000                      )
    declare_property(self, kw, 'n_critic'            , 0                         )
    declare_property(self, kw, 'save_interval'       , 9                         ) 
    declare_property(self, kw, 'use_gradient_penalty', True                      )
    declare_property(self, kw, 'grad_weight'         , 10.0                      )
    declare_property(self, kw, 'gen_optimizer'       , tf.optimizers.Adam(lr=1e-4, beta_1=0.5, decay=1e-4 )     )
    declare_property(self, kw, 'critic_optimizer'    , tf.optimizers.Adam(lr=1e-4, beta_1=0.5, decay=1e-4 )     )
    declare_property(self, kw, 'disp_for_each'       , 0                         )
    declare_property(self, kw, 'save_for_each'       , 0                         )
    declare_property(self, kw, 'output_dir'          , 'output'                  )
    declare_property(self, kw, 'notebook'            , True                      )
    declare_property(self, kw, 'history'             , None                      ) # in case of panic
    declare_property(self, kw, 'start_from_epoch'    , 0                         ) # in case of panic

    # Initialize critic and generator networks
    self.critic        = critic
    self.generator     = generator
    self.latent_dim    = generator.layers[0].input_shape[0][1]
    self.height        = critic.layers[0].input_shape[0][1]
    self.width         = critic.layers[0].input_shape[0][2]
  
    


 

  #
  # Train models
  #
  def fit(self, train_generator, val_generator = None ):


    output = os.getcwd()+'/'+self.output_dir
    if not os.path.exists(output): os.makedirs(output)

    local_vars = ['train_critic_loss', 'train_gen_loss', 'train_reg_loss']
    if val_generator:
      local_vars.extend( ['val_critic_loss', 'val_gen_loss'])


    if not self.history:
      # initialize the history for the first time
      self.history = { key:[] for key in local_vars}
    # if not, probably the optimizer will start from epoch x because a shut down

    for epoch in range(self.start_from_epoch, self.max_epochs):

      batches = 0
      _history = { key:[] for key in local_vars}
    
      #
      # Loop over epochs
      #
      for train_real_samples , _ in tqdm( train_generator , desc= 'training: ', ncols=60): 

        if self.n_critic and not ( (batches % self.n_critic)==0 ):
          # Update only critic using train dataset
          train_critic_loss, train_gen_loss, train_reg_loss, train_real_output, train_fake_samples, train_fake_output = self.train_critic(train_real_samples) 
        else:
          # Update critic and generator
          train_critic_loss, train_gen_loss, train_reg_loss, train_real_output, train_fake_samples, train_fake_output = self.train_critic_and_gen(train_real_samples)
        

        if val_generator:
          
          val_real_samples , _ = val_generator.next()

          # calculate val dataset
          val_fake_samples = self.generate( val_real_samples.shape[0] )
          val_real_output, val_fake_output = self.calculate_critic_output( val_real_samples, val_fake_samples )

          # calculate val losses
          val_critic_loss, _ = self.calculate_critic_loss( val_real_samples, val_fake_samples, val_real_output, val_fake_output)
          val_gen_loss = self.calculate_gen_loss( val_fake_samples, val_fake_output ) 



        batches += 1

        # register all local variables into the history
        for key in local_vars:
          _history[key].append(eval(key))

        # stop after n batches
        if batches > len(train_generator):
          break
        
        
        # end of batch

      # get mean for all
      for key in _history.keys():
        self.history[key].append(float(np.mean( _history[key] ))) # to float to be serializable

      
      perc = np.around(100*epoch/self.max_epochs, decimals=1)
      print('Epoch: %i. Training %1.1f%% complete. critic_loss: %.3f. val_critic_loss: %.3f. gen_loss: %.3f. val_gen_loss: %.3f.'
               % (epoch, perc, self.history['train_critic_loss'][-1], self.history['val_critic_loss'][-1], 
                               self.history['train_gen_loss'][-1]  , self.history['val_gen_loss'][-1],
                                ))


      if self.disp_for_each and ( (epoch % self.disp_for_each)==0 ):
        self.display_images(epoch, output)

      # in case of panic, save it
      if self.save_for_each and ( (epoch % self.save_for_each)==0 ):
        self.critic.save(output+'/critic_epoch_%d.h5'%epoch)
        self.generator.save(output+'/generator_epoch_%d.h5'%epoch)
        with open(output+'/history_epoch_%d.json'%epoch, 'w') as handle:
          json.dump(self.history, handle,indent=4)



    return self.history



  #
  # x = real sample, x_hat = generated sample
  #
  def gradient_penalty(self, x, x_hat): 
    batch_size = x.shape[0]
    # 0.0 <= epsilon <= 1.0
    epsilon = tf.random.uniform((batch_size, self.height, self.width, 1), 0.0, 1.0) 
    # Google Search - u_hat is a randomly weighted average between a real and generated sample
    u_hat = epsilon * x + (1 - epsilon) * x_hat 
    with tf.GradientTape() as penalty_tape:
      penalty_tape.watch(u_hat)
      func = self.critic(u_hat)
    grads = penalty_tape.gradient(func, u_hat) #func gradient at u_hat
    norm_grads = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    # regularizer - to avoid overfitting
    regularizer = tf.math.square( tf.reduce_mean((norm_grads - 1) ) ) 
    return regularizer


  #
  # Calculate critic output
  #
  def calculate_critic_output( self, real_samples, fake_samples ):
    # calculate critic outputs
    real_output = self.critic(real_samples)
    fake_output = self.critic(fake_samples)
    return real_output, fake_output


  #
  # Diff between critic output on real instances(real data) and fake instances(generator data)
  #
  def wasserstein_loss(self, y_true, y_pred): 
    return tf.reduce_mean(y_true) - tf.reduce_mean(y_pred)

  #
  # Calculate the critic loss
  #
  def calculate_critic_loss( self, real_samples, fake_samples, real_output, fake_output ):
    grad_regularizer_loss = tf.multiply(tf.constant(self.grad_weight), self.gradient_penalty(real_samples, fake_samples)) if self.use_gradient_penalty else 0
    critic_loss = tf.add( self.wasserstein_loss(real_output, fake_output), grad_regularizer_loss )
    return critic_loss, grad_regularizer_loss

  #
  # Calculate the generator loss
  #
  def calculate_gen_loss( self, fake_samples, fake_output ):
    gen_loss = tf.reduce_mean(fake_output)
    return gen_loss


  def critic_update( self, critic_tape, critic_loss ): #Update critic
    critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
    self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))


  def gen_update( self, gen_tape, gen_loss): #Update generator
    gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
    self.gen_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))


  #
  # critic = critic; Tries to distinguish real data from the data created by the generator
  #
  def train_critic(self, real_samples, update=True): 
    
    with tf.GradientTape() as critic_tape:
      batch_size = real_samples.shape[0]
      fake_samples = self.generate( batch_size )
      # real_output => output from real samples;fake_output => output from fake samples
      real_output, fake_output = self.calculate_critic_output( real_samples, fake_samples ) 
      # critic loss => wasserstein loss between real output and fake output + regularizer
      critic_loss, grad_regularizer_loss = self.calculate_critic_loss( real_samples, fake_samples, real_output, fake_output) 
      gen_loss = self.calculate_gen_loss( fake_samples, fake_output ) 

    if update:
      # critic_tape
      # Backpropagation(negative feedback??) to improve weights of the critic?
      self.critic_update( critic_tape, critic_loss ) 
    
    return critic_loss, gen_loss, grad_regularizer_loss, real_output, fake_samples, fake_output


  #
  #
  #
  def train_critic_and_gen(self, real_samples, update=True):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as critic_tape:
      batch_size = real_samples.shape[0]
      fake_samples = self.generate( batch_size )
      real_output, fake_output = self.calculate_critic_output( real_samples, fake_samples )
      critic_loss, grad_regularizer_loss = self.calculate_critic_loss( real_samples, fake_samples, real_output, fake_output)
      # gen_loss => Variable to improve the generator to try to make critic classify its fake samples as real;
      gen_loss = self.calculate_gen_loss( fake_samples, fake_output ) 
    
    if update:
      # gen_tape, critic_tape
      self.critic_update( critic_tape, critic_loss )
      self.gen_update( gen_tape, gen_loss )

    return critic_loss, gen_loss, grad_regularizer_loss, real_output, fake_samples, fake_output


  def generate(self, nsamples):
    z = tf.random.normal( (nsamples, self.latent_dim) )
    return self.generator( z )



  def display_images(self, epoch, output):
    # disp plot
    fake_samples = self.generate(25)
    fig = plt.figure(figsize=(10, 10))
    for i in range(25):
       plt.subplot(5,5,1+i)
       plt.axis('off')
       plt.imshow(fake_samples[i],cmap='gray')
    if self.notebook:
      plt.show()
    fig.savefig(output + '/fake_samples_epoch_%d.pdf'%epoch)


  def display_hists( self, epoch, output, real_output, fake_output, bins=50):
    fig = plt.figure(figsize=(10, 5))
    kws = dict(histtype= "stepfilled",alpha= 0.5, linewidth = 2)
    plt.hist(real_output , bins = bins, label='real_output', color='b', **kws)
    plt.hist(fake_output , bins = bins, label='fake_output', color='r', **kws)
    #plt.hist2d(real_output, fake_output, bins=50)
    plt.xlabel('Critic Output',fontsize=18,loc='right')
    plt.ylabel('Count',fontsize=18,loc='top')
    plt.yscale('log')
    plt.legend()
    if self.notebook:
      plt.show()
    fig.savefig(output + '/critic_outputs_epoch_%d.pdf'%epoch)

