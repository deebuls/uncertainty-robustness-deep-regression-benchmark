import argparse
import cv2
import h5py
import numpy as np
import os
import time
import tensorflow as tf

import edl
import data_loader
import models
import trainers
import platform
print ("Hostname : ",platform.node())



parser = argparse.ArgumentParser()
parser.add_argument("--model", default="evidential", type=str,
                    choices=["evidential", "dropout", "ensemble", "gaussian", "laplace"])
parser.add_argument("--batch-size", default=32, type=int)
parser.add_argument("--iters", default=60000, type=int)
parser.add_argument("--learning-rate", default=5e-5, type=float)
parser.add_argument('--noisy-data', dest='noisy_data', action='store_true', help='Bool type')
parser.add_argument('--clean-data', dest='noisy_data', action='store_false', help='Bool type')

args = parser.parse_args()
print ('####### trying gpu')
### Try to limit GPU memory to fit ensembles on RTX 2080Ti
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


gpus = tf.config.experimental.list_physical_devices('GPU')
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#if gpus:
#    try:
#        tf.config.experimental.set_virtual_device_configuration(
#            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(
#                memory_limit=7000)])
#    except RuntimeError as e:
#        print ("################GPU ALLOCATION ERROR#########################")
#        print(e)
print ("GOT GPU now loading data , ", gpus)

print ("Loading data Noisy Flag is ", args.noisy_data)
#print ("GOT logocal cpus , ", logical_gpus)
### Load the data
(x_train, y_train), (x_test, y_test) = data_loader.load_depth(noisy=args.noisy_data)
### Create the trainer
if args.model == "evidential":
    trainer_obj = trainers.Evidential
    model_generator = models.get_correct_model(dataset="depth", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=x_train.shape[1:])
    trainer = trainer_obj(model, opts, "depth", args.learning_rate, lam=2e-1, epsilon=0., maxi_rate=0., noisy_data=args.noisy_data)

elif args.model == "dropout":
    trainer_obj = trainers.Dropout
    model_generator = models.get_correct_model(dataset="depth", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=x_train.shape[1:], sigma=False)
    trainer = trainer_obj(model, opts, "depth", args.learning_rate) #check the parameter not correct 

elif args.model == "ensemble":
    trainer_obj = trainers.Ensemble
    model_generator = models.get_correct_model(dataset="depth", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=x_train.shape[1:], sigma=False)
    trainer = trainer_obj(model, opts, "depth", args.learning_rate) #check the parameter not correct 

elif args.model == "gaussian" or args.model == "laplace":
    trainer_obj = trainers.Likelihood
    model_generator = models.get_correct_model(dataset="depth", trainer=trainer_obj)
    model, opts = model_generator.create(input_shape=x_train.shape[1:], sigma=True)
    trainer = trainer_obj(model, opts, args.model, "depth", args.learning_rate, noisy_data=args.noisy_data)

print("Tesnorflow version :",tf.__version__)
### Train the model
model, rmse, nll = trainer.train(x_train, y_train, x_test, y_test, np.array([[1.]]), iters=args.iters, batch_size=args.batch_size, verbose=True)
tf.keras.backend.clear_session()

print("Done training!")
