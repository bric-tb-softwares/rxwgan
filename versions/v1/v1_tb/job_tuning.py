#!/usr/bin/env python3
try:
  from tensorflow.compat.v1 import ConfigProto
  from tensorflow.compat.v1 import InteractiveSession
  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = InteractiveSession(config=config)
except Exception as e:
  print(e)
  print("Not possible to set gpu allow growth")


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

parser.add_argument('-v','--volume', action='store', 
    dest='volume', required = False,
    help = "volume path")

parser.add_argument('-i','--input', action='store', 
    dest='input', required = True, default = None, 
    help = "Input image directory.")

parser.add_argument('-j','--job', action='store', 
    dest='job', required = True, default = None, 
    help = "job configuration.")




#
# Train parameters
#

parser.add_argument('--batch_size', action='store', 
    dest='batch_size', required = False, default = 32, type=int,
    help = "train batch_size")

parser.add_argument('--epochs', action='store', 
    dest='epochs', required = False, default = 1000, type=int,
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

def lock_as_completed_job(output):
  with open(output+'/.complete','w') as f:
    f.write('complete')

def lock_as_failed_job(output):
  with open(output+'/.failed','w') as f:
    f.write('failed')

try:
    job  = json.load(open(args.job, 'r'))
    sort = job['sort']
    target = 1 # tb active
    test = job['test']

    output_dir = args.volume + '/test_%d_sort_%d'%(test,sort)

    #
    # Check if we need to recover something...
    #
    if os.path.exists(output_dir+'/recover.json'):
        print('Enable recover mode.')
        recover = json.load(open(output_dir+'/recover.json', 'r'))
        history = json.load(open(recover['history'], 'r'))
        critic = tf.keras.models.load_model(recover['critic'])
        generator = tf.keras.models.load_model(recover['generator'])
        start_from_epoch = recover['epoch'] + 1
        print('starts from %d epoch...'%start_from_epoch)
    else:
        start_from_epoch= 0
        # create models
        critic = Critic_v1().model
        generator = Generator_v1().model
        history = None

    height = critic.layers[0].input_shape[0][1]
    width  = critic.layers[0].input_shape[0][2]


    dataframe = pd.read_csv(args.input)


    splits = stratified_train_val_test_splits(dataframe,args.seed)[test]
    training_data   = dataframe.iloc[splits[sort][0]]
    validation_data = dataframe.iloc[splits[sort][1]]

    training_data = training_data.loc[training_data.target==target]
    validation_data = validation_data.loc[validation_data.target==target]

    extra_d = {'sort' : sort, 'test':test, 'target':target, 'seed':args.seed}

    # image generator
    datagen = ImageDataGenerator( rescale=1./255 )


    train_generator = datagen.flow_from_dataframe(training_data, directory = None,
                                                  x_col = 'raw_image_path', 
                                                  y_col = 'target',
                                                  batch_size = args.batch_size,
                                                  target_size = (height,width), 
                                                  class_mode = 'raw', 
                                                  shuffle = True,
                                                  color_mode = 'grayscale')

    val_generator   = datagen.flow_from_dataframe(validation_data, directory = None,
                                                  x_col = 'raw_image_path', 
                                                  y_col = 'target',
                                                  batch_size = args.batch_size,
                                                  class_mode = 'raw',
                                                  target_size = (height,width),
                                                  shuffle = True,
                                                  color_mode = 'grayscale')




    #
    # Create optimizer
    #

    is_test = os.getenv('LOCAL_TEST')

    optimizer = wgangp_optimizer( critic, generator, 
                                  n_discr = args.n_discr,
                                  history = history,
                                  start_from_epoch = 0 if is_test else start_from_epoch,
                                  max_epochs = 1 if is_test else args.epochs, 
                                  output_dir = output_dir,
                                  disp_for_each = args.disp_for_each, 
                                  save_for_each = args.save_for_each )


    # Run!
    history = optimizer.fit( train_generator , val_generator, extra_d=extra_d )

    # in the end, save all by hand
    critic.save(output_dir + '/critic_trained.h5')
    generator.save(output_dir + '/generator_trained.h5')
    with open(output_dir+'/history.json', 'w') as handle:
      json.dump(history, handle,indent=4)

    # necessary to work on orchestra
    lock_as_completed_job(args.volume if args.volume else '.')
    sys.exit(0)

except  Exception as e:
    print(e)
    # necessary to work on orchestra
    lock_as_failed_job(args.volume if args.volume else '.')
    sys.exit(1)