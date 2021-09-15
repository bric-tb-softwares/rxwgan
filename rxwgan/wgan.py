
import logging
import os
import sys
import PIL
import numpy as np
import tensorflow
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp




class WGAN(object):

  def __init__(self, **kw):
    #Defining variables of the class
    self.images = []
    self._height               = retrieve_kw(kw, 'height',               128                                                      )
    self._width                = retrieve_kw(kw, 'width',                128                                                      )
    self._max_epochs           = retrieve_kw(kw, 'max_epochs',           1000                                                     )
    self._batch_size           = retrieve_kw(kw, 'batch_size',           3                                                        ) # 32 Originalmente
    self._n_features           = retrieve_kw(kw, 'n_features',           NotSet                                                   )
    self._n_critic             = retrieve_kw(kw, 'n_critic',               0                                                      )
    self._result_file          = retrieve_kw(kw, 'result_file',          "check_file"                                             )
    self._save_interval        = retrieve_kw(kw, 'save_interval',        9                                                        ) # 100 Originalmente
    self._use_gradient_penalty = retrieve_kw(kw, 'use_gradient_penalty', True                                                     )
    self._verbose              = retrieve_kw(kw, 'verbose',              True                                                     )
    self._gen_opt              = retrieve_kw(kw, 'gen_opt',              tf.optimizers.Adam(lr=1e-4, beta_1=0.5, decay=1e-4 )     )
    self._critic_opt           = retrieve_kw(kw, 'critic_opt',           tf.optimizers.Adam(lr=1e-4, beta_1=0.5, decay=1e-4 )     )
    self._tf_call_kw           = retrieve_kw(kw, 'tf_call_kw',           {}                                                       )
    self._grad_weight          = tf.constant( retrieve_kw(kw, 'grad_weight',          10.0                                      ) )
    self._latent_dim           = tf.constant( retrieve_kw(kw, 'latent_dim',           100                                       ) )
    self._leaky_relu_alpha     = retrieve_kw(kw, 'leaky_relu_alpha',     0.3                                                    )

    # Initialize discriminator and generator networks
    self.critic = self._build_critic()
    self.generator = self._build_generator()

  def latent_dim(self):
    return self._latent_dim

  @tf.function
  def latent_log_prob(self, latent): # Prior probability distribution?? (5.6 - Bayesian Statistics)
    prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self._latent_dim),
                                                     scale_diag=tf.ones(self._latent_dim))
    return prior.log_prob(latent)

  @tf.function
  def wasserstein_loss(self, y_true, y_pred): # Diff between critic output on real instances(real data) and fake instances(generator data)
    return tf.reduce_mean(y_true) - tf.reduce_mean(y_pred)

  @tf.function
  def sample_latent_data(self, nsamples): #Used as input for the generator
    return tf.random.normal((nsamples, self._latent_dim))

  @tf.function
  def transform(self, latent):
    return self.generator( latent, **self._tf_call_kw)

  @tf.function
  def generate(self, nsamples):
    return self.transform( self.sample_latent_data( nsamples ))

  def train(self, train_data, name_file): #Method to train the model
    if self._n_features is NotSet:
      self._n_features = train_data.shape[1]
    if self._verbose: print('Number of features is %d.' % self._n_features )
    gpus = tf.config.experimental.list_physical_devices('GPU')
    n_gpus = len(gpus)
    if self._verbose: print('This machine has %i GPUs.' % n_gpus)

    # Getting the training dataset from train_data
    train_dataset = tf.data.Dataset.from_tensor_slices( train_data ).batch( self._batch_size, drop_remainder = True )

    # checkpoint for the model - store info in case of system failure
    checkpoint_maker = tf.train.Checkpoint(generator_optimizer=self._gen_opt,
        discriminator_optimizer=self._critic_opt,
        generator=self.generator,
        discriminator=self.critic
    )if self._result_file else None

    # containers for losses
    losses = {'critic': [], 'generator': [], 'regularizer': []}
    critic_acc = []

    #reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
    #                              factor=factor, cooldown=0, min_lr=1e-4, verbose=2) Reduce Learning Rate when a metric has stopped improving
    #model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
    #          class_weight=class_weight, verbose=2, validation_data=(X_test, y_test)) Train model for a fixed number of epochs

    updates = 0; batches = 0;
    for epoch in range(self._max_epochs):
      for sample_batch in train_dataset:
        if self._n_critic and (updates % self._n_critic):
          # Update only critic
          critic_loss, reg_loss, gen_loss = self._train_critic(sample_batch) + (np.nan,)
        if not(self._n_critic) or not(updates % self._n_critic):
          # Update critic and generator
          critic_loss, gen_loss, reg_loss = self._train_step(sample_batch)
        losses['critic'].append(critic_loss)
        losses['generator'].append(gen_loss)
        losses['regularizer'].append(reg_loss)
        updates += 1
        #perc = np.around(100*epoch/self._max_epochs, decimals=1)

        # Save current model
        #if checkpoint_maker and not(updates % self._save_interval):
        #  checkpoint_maker.save(file_prefix=self._result_file)
        #  pass
        # Print logging information
        if self._verbose and not(updates % self._save_interval):
          perc = np.around(100*epoch/self._max_epochs, decimals=1)
          print('Epoch: %i. Updates %i. Training %1.1f%% complete. Critic_loss: %.3f. Gen_loss: %.3f. Regularizer: %.3f'
               % (epoch, updates, perc, critic_loss, gen_loss, reg_loss ))

    checkpoint_maker.save(file_prefix=name_file) # Save checkpoint file
    self.save( name_file, True )
    return losses

  def save(self, name_file, overwrite = False ): # Save generator and critic weights
    self.generator.save_weights( "./" + name_file + "/" + name_file + '_generator', overwrite )
    self.critic.save_weights( "./" + name_file + "/" + name_file + '_critic', overwrite )

  def load(self, path ): # Load generator and critic weights
    self.generator.load_weights( path + '_generator' )
    self.critic.load_weights( path + '_critic' )

  def _build_critic(self):
    ip = layers.Input(shape=(self._height,self._width,1))
    # TODO Add other normalization scheme as mentioned in the article
    # Input (None, 3^2*2^5 = 1 day = 288 samples, 1)
    #padding = 'same' -> new.height = in.height/strides.row
    #padding = 'valid' -> new.height = (in.height - kernel.height + 1)/strides.row
    y = layers.Conv2D(256, (5,5), strides=(2,2), padding='same', kernel_initializer='he_uniform', data_format='channels_last', input_shape=(self._height,self._width,1))(ip)
    #y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y) #Rectified linear activation function -> output = 0 for neg/zero inputs else output = input
    y = layers.Dropout(rate=0.3, seed=1)(y) #randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting
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
    #y = layers.Conv2D(32, 5, strides=2, padding='same', kernel_initializer='he_uniform')(y)
    #y = layers.Activation('relu')(y)
    #y = layers.Dropout(rate=0.3, seed=1)(y)
    #Output (None, 8, 8, 32)
    y = layers.Flatten()(y)
    # Output (None, 3*256)
    #out = layers.Dense(nb_class, activation='sigmoid')(y)
    out = layers.Dense(1, activation='linear')(y) #activation='linear' returns input unmodified
    # Output (None, 1)
    model = tf.keras.Model(ip, out)
    if self._verbose: model.summary()
    model.compile()
    #y = layers.GlobalAveragePooling1D()(y)
    return model



  def _build_generator(self):
    ip = layers.Input(shape=(self._latent_dim,))
    # Input (None, latent space (100?) )
    y = layers.Dense(units=16*16*32, input_shape=(self._latent_dim,))(ip)
    # Output (None, 64*3^2 )
    y = layers.Reshape(target_shape=(16,16, 32))(y)
    #y = layers.BatchNormalization()(y)
    #y = layers.LeakyReLU(alpha=self._leaky_relu_alpha)(y)
    #y = layers.UpSampling1D()(y)
    # Output (None, 3^2*2, 64)
    y = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(alpha=self._leaky_relu_alpha)(y)
    y = layers.Dropout(rate=0.3)(y)
    #y = layers.UpSampling1D(size=2*2)(y)
    # Output (None, 3^2*2^3, 128)
    y = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(alpha=self._leaky_relu_alpha)(y)
    y = layers.Dropout(rate=0.3)(y)
    #y = layers.UpSampling1D(size=2*2)(y)
    # Output (None, 3^2*2^5, 256)
    y = layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(alpha=self._leaky_relu_alpha)(y)
    y = layers.Dropout(rate=0.3)(y)
    # Output (None, 3^2*2^5, 64)
    y = layers.Conv2DTranspose(512, (4,4), strides=(1,1), padding='same', kernel_initializer='he_uniform')(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(alpha=self._leaky_relu_alpha)(y)
    y = layers.Dropout(rate=0.3)(y)
    # Output (None, 128, 128, 512)
    out = layers.Conv2DTranspose(1, (4,4), strides=(1,1), padding='same', kernel_initializer='he_uniform', activation = 'tanh')(y)
    # Output (None, 3^2*2^5, 1)
    model = tf.keras.Model(ip, out)
    if self._verbose: model.summary()
    model.compile()
    return model

  @tf.function
  def _gradient_penalty(self, x, x_hat): #x = real sample; x_hat = generated sample
    epsilon = tf.random.uniform((self._batch_size, self._height, self._width, 1), 0.0, 1.0) # 0.0 <= epsilon <= 1.0
    u_hat = epsilon * x + (1 - epsilon) * x_hat #Google Search - u_hat is a randomly weighted average between a real and generated sample
    with tf.GradientTape() as penalty_tape:
      penalty_tape.watch(u_hat)
      func = self.critic(u_hat)
    grads = penalty_tape.gradient(func, u_hat) #func gradient at u_hat
    norm_grads = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    regularizer = tf.math.square( tf.reduce_mean((norm_grads - 1) ) ) #regularizer - to avoid overfitting
    return regularizer


  @tf.function
  def _get_critic_output( self, samples, fake_samples ):
    # calculate critic outputs
    real_output = self.critic(samples, **self._tf_call_kw)
    fake_output = self.critic(fake_samples, **self._tf_call_kw)
    return real_output, fake_output

  @tf.function
  def _get_critic_loss( self, samples, fake_samples, real_output, fake_output ):
    grad_regularizer_loss = tf.multiply(self._grad_weight, self._gradient_penalty(samples, fake_samples)) if self._use_gradient_penalty else 0
    critic_loss = tf.add( self.wasserstein_loss(real_output, fake_output), grad_regularizer_loss )
    return critic_loss, grad_regularizer_loss

  def _get_gen_loss( self, fake_samples, fake_output ):
    gen_loss = tf.reduce_mean(fake_output)
    return gen_loss

  def _apply_critic_update( self, critic_tape, critic_loss ): #Update Critic
    critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
    self._critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
    return

  def _apply_gen_update( self, gen_tape, gen_loss): #Update generator
    gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
    self._gen_opt.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
    return

  @tf.function
  def _train_critic(self, samples): # Critic = Discriminator; Tries to distinguish real data from the data created by the generator
    with tf.GradientTape() as critic_tape:
      fake_samples = self.generate( self._batch_size )
      real_output, fake_output = self._get_critic_output( samples, fake_samples ) #real_output => output from real samples;fake_output => output from fake samples
      critic_loss, grad_regularizer_loss = self._get_critic_loss( samples, fake_samples, real_output, fake_output) #critic loss => wasserstein loss between real output and fake output + regularizer
    # critic_tape
    self._apply_critic_update( critic_tape, critic_loss ) # Backpropagation(negative feedback??) to improve weights of the critic?
    return critic_loss, grad_regularizer_loss

  @tf.function
  def _train_step(self, samples):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as critic_tape:
      fake_samples = self.generate( self._batch_size )
      real_output, fake_output = self._get_critic_output( samples, fake_samples )
      critic_loss, critic_regularizer = self._get_critic_loss( samples, fake_samples, real_output, fake_output)
      gen_loss = self._get_gen_loss( fake_samples, fake_output ) #gen_loss => Variable to improve the generator to try to make critic classify its fake samples as real;
    # gen_tape, critic_tape
    self._apply_critic_update( critic_tape, critic_loss )
    self._apply_gen_update( gen_tape, gen_loss )
    return critic_loss, gen_loss, critic_regularizer
