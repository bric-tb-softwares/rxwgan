#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from rxwgan.wgangp import wgangp_optimizer
#from rxwgan.kfold import KFold
from sklearn.model_selection import KFold


import tensorflow as tf  
import json

parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()

parser.add_argument('-o','--output', action='store', 
    dest='output', required = False,
    help = "Output dir")

parser.add_argument('-i','--input', action='store', 
    dest='input', required = True, default = None, 
    help = "Input image directory.")

parser.add_argument('-g','--generator', action='store', 
    dest='generator', required = True,
    help = "The path to the h5 generator model.")

parser.add_argument('-c','--critic', action='store', 
    dest='critic', required = True,
    help = "The path to the h5 critic model.")


#
# Train parameters
#

parser.add_argument('--batch_size', action='store', 
    dest='batch_size', required = False, default = 32, type=int,
    help = "train batch_size")

parser.add_argument('--epochs', action='store', 
    dest='epochs', required = False, default = 500, type=int,
    help = "Number of epochs.")

parser.add_argument('--n_discr', action='store', 
    dest='n_discr', required = False, default = 0, type=int, 
    help = "Update the discriminator after n batches.")

parser.add_argument('--save_for_each', action='store', 
    dest='save_for_each', required = False, default = 50, type=int, 
    help = "Save model after N epochs.")

parser.add_argument('--disp_for_each', action='store', 
    dest='disp_for_each', required = False, default = 50, type=int, 
    help = "Save plots after N epochs.")

#
# K Fold method parameters
#

parser.add_argument('--n_splits', action='store', 
    dest='n_splits', required = False, default = 5, type=int, 
    help = "How many k folds you will have at the end?")

parser.add_argument('-f', '--fold', action='store', 
    dest='fold', required = False, default = 0, type=int, 
    help = "The current fold number to process")

parser.add_argument('--seed', action='store', 
    dest='seed', required = False, default = 512, type=int, 
    help = "Seed value to initialize the k fold stage.")



#
# panic options
#

parser.add_argument('--start_from_epoch', action='store', 
    dest='start_from_epoch', required = False, default = 0, type=int, 
    help = "Use this option to start from the last epoch + 1. You shoud pass as argument history,critic and generator from last epoch.")

parser.add_argument('--history', action='store', 
    dest='history', required = False, default = None, 
    help = "The path of the last history. Should be used with start_from_epoch option.")

import sys,os
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

args = parser.parse_args()

# load generator
generator = tf.keras.models.load_model(args.generator)
generator.summary()

# load critic
critic = tf.keras.models.load_model(args.critic)
critic.summary()

height = critic.layers[0].input_shape[0][1]
width  = critic.layers[0].input_shape[0][2]

output_dir = 'output/fold_%d'%args.fold

dataframe = pd.read_csv(args.input)


kf = KFold(n_splits = args.n_splits, shuffle = True, random_state = args.seed)
#splits = [(train_index, val_index, test_index) for train_index, val_index, test_index in kf.split(dataframe)]
splits = [(train_index, val_index) for train_index, val_index in kf.split(dataframe)]

training_data   = dataframe.iloc[splits[args.fold][0]]
validation_data = dataframe.iloc[splits[args.fold][1]]

# image generator
datagen = ImageDataGenerator( rescale=1./255 )


train_generator = datagen.flow_from_dataframe(training_data, directory = None,
                                              x_col = 'raw_image_path', 
                                              y_col = 'target',
                                              batch_size = args.batch_size,
                                              target_size = (height,width), 
                                              class_mode = 'raw', 
                                              shuffle = False,
                                              color_mode = 'grayscale')

val_generator   = datagen.flow_from_dataframe(validation_data, directory = None,
                                              x_col = 'raw_image_path', 
                                              y_col = 'target',
                                              batch_size = args.batch_size,
                                              class_mode = 'raw',
                                              target_size = (height,width),
                                              shuffle = False,
                                              color_mode = 'grayscale')

#
# Create optimizer
#

if args.history:
    history = json.load(open(args.history, 'r'))
else:
    history = None


optimizer = wgangp_optimizer( critic, generator, 
                              n_discr = args.n_discr, 
                              max_epochs = args.epochs, 
                              start_from_epoch = args.start_from_epoch,
                              history = history,
                              output_dir = output_dir,
                              disp_for_each = args.disp_for_each, 
                              save_for_each=args.save_for_each )


# Run!
history = optimizer.fit( train_generator , val_generator )

# in the end, save all by hand
critic.save(output_dir + '/critic_trained.h5')
generator.save(output_dir + '/generator_trained.h5')
with open(output_dir+'/history.json', 'w') as handle:
  json.dump(history, handle,indent=4)

