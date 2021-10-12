#!/usr/bin/env python3
import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from rxwgan.wgan import wgan
import tensorflow as tf


parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()

parser.add_argument('-o','--output', action='store', 
    dest='output', required = False,
    help = "Output dir")

parser.add_argument('-i','--input', action='store', 
    dest='input', required = True, default = None, 
    help = "Input image directory.")

parser.add_argument('-v','--version', action='store', 
    dest='version', required = False, default = 1, type=int,
    help = "Version of the model.")



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



import sys,os
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)
args = parser.parse_args()


if args.version == 1:
    from rxwgan.models import Model_v1 as Model
else:
    print('version (%d) not supported.' % args.version )
    sys.exit(1)

# create the model (discr and gen)
gan = Model(trainable=True)


class_name = args.input.split('/')[-1]
path = args.input.replace(class_name,'').replace('//','/')

train_datagen = ImageDataGenerator( rescale=1./255 )
train_generator = train_datagen.flow_from_directory( path, 
                                                     color_mode='grayscale', 
                                                     target_size=(gan.height,gan.width), 
                                                     batch_size=args.batch_size,
                                                     classes=[class_name] 
                                                     )

optimizer = wgan( gan, n_discr=args.n_discr, max_epochs=args.epochs,
                       save_for_each = args.save_for_each, 
                       disp_for_each = args.disp_for_each,
                       output = ars.output_dir,
                       notebook = False)
optimizer.train( train_generator )



sys.exit(0)