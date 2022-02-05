
__all__ = ['wgangp_optimizer']

import logging
import numpy as np
import itertools
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json

import atlas_mpl_style as ampl
ampl.use_atlas_style()

from rxcore import declare_property
from rxcore.stats import calculate_divergences, calculate_l1_and_l2_norm_errors, eps

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
    declare_property(self, kw, 'output_dir'          , None                      )
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
  def fit(self, train_generator, val_generator = None, extra_d=None , wandb=None):

    if self.output_dir is not None:
      output = self.output_dir
      if not os.path.exists(output): os.makedirs(output)
    else:
      output = os.getcwd()

    train_local_vars = ['train_critic_loss', 
                        'train_gen_loss', 
                        'train_reg_loss', 
                        'train_kl_rr',
                        'train_kl_rf', 
                        'train_js_rr',
                        'train_js_rf', 
                        'train_l1_rr', 
                        'train_l1_rf',
                        'train_l2_rr', 
                        'train_l2_rf']

    val_local_vars = ['val_critic_loss', 
                      'val_gen_loss', 
                      'val_kl_rr',
                      'val_kl_rf', 
                      'val_js_rr',
                      'val_js_rf', 
                      'val_l1_rr', 
                      'val_l1_rf', 
                      'val_l2_rr', 
                      'val_l2_rf']
    
    if not self.history:
      # initialize the history for the first time
      self.history = { key:[] for key in train_local_vars}
      if val_generator:
        self.history.update({ key:[] for key in val_local_vars})


    # if not, probably the optimizer will start from epoch x because a shut down
    for epoch in range(self.start_from_epoch, self.max_epochs):

      _history = { key:[] for key in self.history.keys()}
    
      #
      # Loop over epochs
      #
      batches = 0
      for train_real_samples , _ in tqdm( train_generator , desc= 'training: ', ncols=60): 

        if self.n_critic and not ( (batches % self.n_critic)==0 ):
          # Update only critic using train dataset
          train_critic_loss, train_gen_loss, train_reg_loss, train_real_output, train_fake_samples, train_fake_output = self.train_critic(train_real_samples) 
        else:
          # Update critic and generator
          train_critic_loss, train_gen_loss, train_reg_loss, train_real_output, train_fake_samples, train_fake_output = self.train_critic_and_gen(train_real_samples)
        
        
        skip_local_vars = []

        # train must have at least two events
        if train_real_samples.shape[0] > 1:

          # calculate divergences
          train_kl_rr, train_js_rr = self.calculate_divergences( train_real_samples, train_real_samples )
          train_kl_rf, train_js_rf = self.calculate_divergences( train_real_samples, train_fake_samples.numpy() )

          # calculate l1 and l2 norm errors
          train_l1_rr, train_l2_rr = self.calculate_l1_and_l2_norm_errors(train_real_samples, train_real_samples )
          train_l1_rf, train_l2_rf = self.calculate_l1_and_l2_norm_errors(train_real_samples, train_fake_samples.numpy() )
        else:
          skip_local_vars = ['train_kl_rr','train_kl_rf', 'train_js_rr','train_js_rf', 
                             'train_l1_rr', 'train_l1_rf','train_l2_rr', 'train_l2_rf']

        batches += 1

        # register all local variables into the history
        for key in train_local_vars:
          if not key in skip_local_vars:
            _history[key].append(eval(key))

        # stop after n batches
        if batches > len(train_generator):
          break

        # end of train batch


      if val_generator:
        batches = 0
        for val_real_samples , _ in tqdm( val_generator , desc= 'validation: ', ncols=60): 

          # calculate val dataset
          val_fake_samples = self.generate( val_real_samples.shape[0] ).numpy()
          val_real_output, val_fake_output = self.calculate_critic_output( val_real_samples, val_fake_samples )
          # calculate val losses
          val_critic_loss, _ = self.calculate_critic_loss( val_real_samples, val_fake_samples, val_real_output, val_fake_output)
          val_gen_loss = self.calculate_gen_loss( val_fake_samples, val_fake_output ) 

          skip_local_vars = []

          # validation must have at least two events
          if val_real_samples.shape[0] > 1:

            # calculate divergences
            val_kl_rr, val_js_rr = self.calculate_divergences( val_real_samples, val_real_samples )
            val_kl_rf, val_js_rf = self.calculate_divergences( val_real_samples, val_fake_samples )

            # calculate l1 and l2 norm errors
            val_l1_rr, val_l2_rr = self.calculate_l1_and_l2_norm_errors(val_real_samples, val_real_samples)
            val_l1_rf, val_l2_rf = self.calculate_l1_and_l2_norm_errors(val_real_samples, val_fake_samples)
          
          else:
            skip_local_vars = ['train_kl_rr','train_kl_rf', 'train_js_rr','train_js_rf', 
                               'train_l1_rr', 'train_l1_rf','train_l2_rr', 'train_l2_rf']

          
          # register all local variables into the history
          for key in val_local_vars:
            if not key in skip_local_vars:
              _history[key].append(eval(key))
  
          batches += 1

          # stop after n batches
          if batches > len(val_generator):
            break

      
        # end of val batch


      # get mean for all
      for key in _history.keys():
        self.history[key].append(float(np.mean( _history[key] ))) # to float to be serializable

      perc = np.around(100*epoch/self.max_epochs, decimals=1)

      if val_generator:
        print('Epoch: %i. Training %1.1f%% complete. critic_loss: %.3f. val_critic_loss: %.3f. gen_loss: %.3f. val_gen_loss: %.3f. val_kl_rf: %.3f. val_js_rf: %.3f.'
               % (epoch, perc, self.history['train_critic_loss'][-1], self.history['val_critic_loss'][-1], 
                               self.history['train_gen_loss'][-1]  , self.history['val_gen_loss'][-1],
                               self.history['val_kl_rf'][-1], self.history['val_js_rf'][-1]
                                ))
      else:
        print('Epoch: %i. Training %1.1f%% complete. critic_loss: %.3f. gen_loss: %.3f.'
               % (epoch, perc, self.history['train_critic_loss'][-1], self.history['train_gen_loss'][-1]))


      if wandb:
        log = {}
        for key in self.history.keys():
          log[key] = self.history[key][-1]
        wandb.log(log)


      if self.disp_for_each and ( (epoch % self.disp_for_each)==0 ):
        self.display_images(epoch, output, wandb=wandb)
       

      # in case of panic, save it
      if self.save_for_each and ( (epoch % self.save_for_each)==0 ):
        self.critic.save(output+'/critic_epoch_%d.h5'%epoch)
        self.generator.save(output+'/generator_epoch_%d.h5'%epoch)
        with open(output+'/history_epoch_%d.json'%epoch, 'w') as handle:
          json.dump(self.history, handle,indent=4)

        with open(output+'/checkpoint.json', 'w') as handle:
          d = {'epoch'     :epoch, 
               'history'   : output+'/history_epoch_%d.json'%epoch,
               'critic'    : output+'/critic_epoch_%d.h5'%epoch,
               'generator' : output+'/generator_epoch_%d.h5'%epoch,
          }

          if extra_d:
            d.update(extra_d)
          json.dump(d, handle,indent=4)

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



  def display_images(self, epoch, output, wandb=None):
    # disp plot
    fake_samples = self.generate(25)
    fig = plt.figure(figsize=(10, 10))
    for i in range(25):
       plt.subplot(5,5,1+i)
       plt.axis('off')
       plt.imshow(fake_samples[i],cmap='gray')
    if self.notebook:
      plt.show()
    figname = output + '/fake_samples_epoch_%d.pdf'%epoch
    fig.savefig(figname)
    if wandb:
      wandb.log({'fake_samples':plt})


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


  def calculate_divergences( self , real_samples, fake_samples ):
    kl, js = calculate_divergences(real_samples, fake_samples)
    return np.mean(kl), np.mean(js)


  def calculate_l1_and_l2_norm_errors(self, real_samples, fake_samples):
    l1, l2 = calculate_l1_and_l2_norm_errors( real_samples, fake_samples )
    return np.mean(l1),  np.mean(l2)



