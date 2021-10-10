#!/usr/bin/env python3
import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from rxwgan.wgan import wgan
import tensorflow as tf


parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()

parser.add_argument('-o','--output', action='store', 
    dest='outputFile', required = False,
    help = "Output file.")

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


train_datagen = ImageDataGenerator( rescale=1./255 )
train_generator = train_datagen.flow_from_directory( args.input, 
                                                     color_mode='grayscale', 
                                                     target_size=(gan.height,gan.width), 
                                                     batch_size=args.batch_size,
                                                     classes=['tb'] 
                                                     )

optimizer = wgan( gan, batch_size = args.batch_size, n_discr=args.n_discr, max_epochs=args.epochs )
optimizer.train( train_generator )


sys.exit(0)
