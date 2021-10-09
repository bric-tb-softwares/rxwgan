



def generator( latent_dim, leaky_relu_alpha = 0.3, verbose=False ):

    ip = layers.Input(shape=(latent_dim,))
    # Input (None, latent space (100?) )
    y = layers.Dense(units=16*16*32, input_shape=(latent_dim,))(ip)
    # Output (None, 64*3^2 )
    y = layers.Reshape(target_shape=(16,16, 32))(y)
    #y = layers.BatchNormalization()(y)
    #y = layers.LeakyReLU(alpha=leaky_relu_alpha)(y)
    #y = layers.UpSampling1D()(y)
    # Output (None, 3^2*2, 64)
    y = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(alpha=leaky_relu_alpha)(y)
    y = layers.Dropout(rate=0.3)(y)
    #y = layers.UpSampling1D(size=2*2)(y)
    # Output (None, 3^2*2^3, 128)
    y = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(alpha=leaky_relu_alpha)(y)
    y = layers.Dropout(rate=0.3)(y)
    #y = layers.UpSampling1D(size=2*2)(y)
    # Output (None, 3^2*2^5, 256)
    y = layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(alpha=leaky_relu_alpha)(y)
    y = layers.Dropout(rate=0.3)(y)
    # Output (None, 3^2*2^5, 64)
    out = layers.Conv2DTranspose(1, (4,4), strides=(1,1), padding='same', kernel_initializer='he_uniform', activation = 'tanh')(y)
    # Output (None, 3^2*2^5, 1)
    model = tf.keras.Model(ip, out)
    if verbose: model.summary()
    model.compile()
    return model




def discrinator( height, width ):

    ip = layers.Input(shape=(height,width,1))
    # TODO Add other normalization scheme as mentioned in the article
    # Input (None, 3^2*2^5 = 1 day = 288 samples, 1)
    y = layers.Conv2D(256, (5,5), strides=(2,2), padding='same', kernel_initializer='he_uniform', data_format='channels_last', input_shape=(height,width,1))(ip)
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
    if verbose: model.summary()
    model.compile()
    #y = layers.GlobalAveragePooling1D()(y)
    return model