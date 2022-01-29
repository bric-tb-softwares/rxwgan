#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from rxwgan.wgangp import wgangp_optimizer
from rxwgan.models.models_v1 import *
from rxwgan.stratified_kfold import stratified_train_val_test_splits
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

parser.add_argument('-j','--job', action='store', 
    dest='job', required = True, default = None, 
    help = "job configuration.")

parser.add_argument('--test', action='store', 
    dest='test', required = True, default = None, type=int,
    help = "test configuration.")



#
# Train parameters
#

parser.add_argument('-t', '--target', action='store', 
    dest='target', required = True, default = 1, type=int,
    help = "target: 0/1")


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

parser.add_argument('--seed', action='store', 
    dest='seed', required = False, default = 512, type=int, 
    help = "Seed value to initialize the k fold stage.")


import sys,os
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

args = parser.parse_args()


print(args.input)
job = json.load(open(args.job, 'r'))

sort       = job['sort']


# create models
critic = Critic_v1().model
generator = Generator_v1().model

critic.summary()
generator.summary()

height = critic.layers[0].input_shape[0][1]
width  = critic.layers[0].input_shape[0][2]

output_dir = 'output/sort_%d'%(sort)

dataframe = pd.read_csv(args.input)


splits = stratified_train_val_test_splits(dataframe)[args.test]
training_data   = dataframe.iloc[splits[sort][0]]
validation_data = dataframe.iloc[splits[sort][1]]

target=args.target
training_data = training_data.loc[training_data.target==args.target]
validation_data = validation_data.loc[validation_data.target==args.target]


print(dataframe.shape)
print(training_data.shape)
print(validation_data.shape)

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


optimizer = wgangp_optimizer( critic, generator, 
                              n_discr = args.n_discr, 
                              max_epochs = args.epochs, 
                              #start_from_epoch = args.start_from_epoch,
                              #history = history,
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

