#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for training colorize GAN. 

@author: khe
"""
import model
import datautils
import tensorflow as tf
import numpy as np
import argparse
import logging
from datetime import datetime
import os

# Set up logs
os.makedirs('logs', exist_ok=True)
today = datetime.utcnow()
logging.basicConfig(format='%(asctime)s %(message)s',
                    filename='logs/%s.log'%(today.isoformat()[:10].replace('-', '_')),
                    level=logging.INFO)

# Model hyperparams and constants
GEN_LR = 5e-4
DISC_LR = 1e-4
DISC_CV = 0.1
EPOCHS = 100
BATCH_SIZE = 20
IMG_HEIGHT = 90
IMG_WIDTH = 120
Z_DIM = 100

# Get model instances
generator = model.make_generator_model(IMG_HEIGHT, IMG_WIDTH, Z_DIM)
discriminator = model.make_discriminator_model(IMG_HEIGHT, IMG_WIDTH)

generator_optimizer = model.generator_optimizer(GEN_LR)
discriminator_optimizer = model.discriminator_optimizer(DISC_LR, DISC_CV)

@tf.function
def train_step(train_data):
    '''
    Custom training function for mini-batch.

    Parameters
    ----------
    train_data : tensor

    Returns
    -------
    gen_loss : tensor
        Generator loss.
    disc_loss : tensor
        Discriminator loss. 

    '''
    L = train_data[:,:,:,:1]
    ab = train_data[:,:,:,1:]
    z = tf.random.uniform((L.shape[0], Z_DIM),-1,1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_ab = generator([L,z], training=False)

        logits_real = discriminator(ab, training=False)
        logits_fake = discriminator(generated_ab, training=False)

        gen_loss = model.generator_loss(ab, generated_ab)
        disc_loss = model.discriminator_loss(logits_real, logits_fake)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss
            
def train(epochs):
    '''Main train loop.

    '''
    for epoch in range(epochs):
        counter = 0
        total_gen_loss = 0
        total_disc_loss = 0
        
        # Shuffle data
        chunks = np.arange(0, 15900, 500)
        chunks = np.concatenate((chunks, [15900]))
        for i in range(len(chunks)-1):
            this_data = datautils.load_training_data('data/youtube_data.h5', 
                                                     chunks[i], chunks[i+1],
                                                     IMG_HEIGHT, IMG_WIDTH)
            idx = np.arange(this_data.shape[0])
            np.random.shuffle(idx)
            for j in range(0, idx.shape[0], BATCH_SIZE):
                this_idx = idx[j:j+BATCH_SIZE]
                train_data = tf.cast(this_data[this_idx], tf.float32)
                gen_loss, disc_loss = train_step(train_data)
                
                counter += 1
                total_gen_loss += gen_loss
                total_disc_loss += disc_loss
                
        logging.info('Epoch: %d, generator loss: %.6f, discriminator loss: %.6f'%(epoch+1, 
                      total_gen_loss/counter, total_disc_loss/counter))                 

        if (epoch + 1) % 10 == 0:
            # Save model 
            generator.save_weights('data/generator/weights')
            discriminator.save_weights('data/discriminator/weights')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build_db', dest='build_db', 
                        help='Build database', action='store_true')
    parser.add_argument('--load_weights', dest='load_weights', 
                        help='Load model weights', action='store_true')
    parser.add_argument('--epoch', dest='epoch', help='Number of epochs', 
                        default=None, required=False, type=int)
    args = parser.parse_args()
    
    if args.build_db:
        datautils.build_database()
    
    if args.load_weights:
        generator.load_weights('data/generator/weights')
        discriminator.load_weights('data/discriminator/weights')
        
    if args.epoch is None:
        epochs = EPOCHS
    else:
        epochs = args.epoch
        
    train(epochs)