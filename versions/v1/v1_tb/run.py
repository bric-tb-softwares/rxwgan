#!/usr/bin/env python3


import pandas as pd
import numpy as np
import argparse, traceback, json, os
from rxwgan.models import *
from rxwgan.wgangp import wgangp_optimizer
from colorama import Back, Fore

from rxcore import stratified_train_val_test_splits
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf 

from rxcore import allow_tf_growth
allow_tf_growth()

# NOTE: for orchestra 
is_test = True if 'LOCAL_TEST' in os.environ.keys() else False

#
# Input args (mandatory!)
#
parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()

parser.add_argument('-v','--volume', action='store', 
    dest='volume', required = False, default=os.getcwd(),
    help = "volume path")

parser.add_argument('-i','--input', action='store', 
    dest='input', required = True, default = None, 
    help = "Input image directory.")

parser.add_argument('-j','--job', action='store', 
    dest='job', required = True, default = None, 
    help = "job configuration.")

parser.add_argument('--disable_wandb', action='store_true', 
    dest='disable_wandb', required = False, 
    help = "Disable wandb report")



import sys,os
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

args = parser.parse_args()


try:


    #
    # Start your job here
    #

    job  = json.load(open(args.job, 'r'))
    sort = job['sort']
    target = 1 # tb active
    test = job['test']
    seed = 512
    epochs = 1000
    batch_size = 8


    #
    # Check if we need to recover something...
    #
    if os.path.exists(args.volume+'/checkpoint.json'):
        print(Back.RED + Fore.WHITE + 'reading from last checkpoint...')

        checkpoint = json.load(open(args.volume+'/checkpoint.json', 'r'))
        history = json.load(open(checkpoint['history'], 'r'))
        critic = tf.keras.models.load_model(checkpoint['critic'])
        generator = tf.keras.models.load_model(checkpoint['generator'])
        start_from_epoch = checkpoint['epoch'] + 1
        print(Back.RED + Fore.WHITE + 'starts from %d epoch...'%start_from_epoch)
    else:
        start_from_epoch= 0
        # create models
        critic = Critic_v1().model
        generator = Generator_v1().model
        history = None

    height = critic.layers[0].input_shape[0][1]
    width  = critic.layers[0].input_shape[0][2]

    # Read dataframe
    dataframe = pd.read_csv(args.input)


    splits = stratified_train_val_test_splits(dataframe,10,seed)[test]
    training_data   = dataframe.iloc[splits[sort][0]]
    validation_data = dataframe.iloc[splits[sort][1]]
    training_data   = training_data.loc[training_data.target==target]
    validation_data = validation_data.loc[validation_data.target==target]

    extra_d = {'sort' : sort, 'test':test, 'target':target, 'seed':seed}

    # image generator
    datagen = ImageDataGenerator( rescale=1./255 )

    train_generator = datagen.flow_from_dataframe(training_data, directory = None,
                                                  x_col = 'raw_image_path', 
                                                  y_col = 'target',
                                                  batch_size = batch_size,
                                                  target_size = (height,width), 
                                                  class_mode = 'raw', 
                                                  shuffle = True,
                                                  color_mode = 'grayscale')

    val_generator   = datagen.flow_from_dataframe(validation_data, directory = None,
                                                  x_col = 'raw_image_path', 
                                                  y_col = 'target',
                                                  batch_size = batch_size,
                                                  class_mode = 'raw',
                                                  target_size = (height,width),
                                                  shuffle = True,
                                                  color_mode = 'grayscale')

    #
    # Create optimizer
    #

    optimizer = wgangp_optimizer( critic, generator, 
                                  n_discr = 0,
                                  history = history,
                                  start_from_epoch = 0 if is_test else start_from_epoch,
                                  max_epochs = 1 if is_test else epochs, 
                                  volume = args.volume,
                                  disp_for_each = 10, 
                                  save_for_each = 200 )

    
    try:
        if args.disable_wandb:
            wandb=None
        else:
            import wandb
            task = args.volume.split('/')[-2]
            name = 'test_%d_sort_%d'%(test,sort)
            wandb.init(project=task,
                       name=name,
                       id=name)

    except:
        wandb=None


    # Run!
    history = optimizer.fit( train_generator , val_generator, extra_d=extra_d, wandb=wandb )

    # in the end, save all by hand
    critic.save(args.volume + '/critic_trained.h5')
    generator.save(args.volume + '/generator_trained.h5')
    with open(args.volume+'/history.json', 'w') as handle:
      json.dump(history, handle,indent=4)

    #
    # End your job here
    #

    sys.exit(0)

except  Exception as e:
    traceback.print_exc()
    sys.exit(1)