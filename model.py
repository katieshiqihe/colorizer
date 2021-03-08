#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colorize GAN model.

@author: khe
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Concatenate, Dense, BatchNormalization, ReLU, LeakyReLU, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE, BinaryCrossentropy

###############################################################################
# Model
###############################################################################

def make_generator_model(img_height, img_width, z_dim):
    '''
    Colorizer generator.

    Parameters
    ----------
    img_height : int
        Input image height.
    img_width : int
        Input image width.
    z_dim : int
        Dimension of noise vector. 

    Returns
    -------
    generator: keras model
        An instance of generator.

    '''
    L = Input((img_height, img_width, 1), name='L')
    z = Input((z_dim), name='z')

    h0 = Dense(img_height*img_width)(z)
    h0 = tf.reshape(h0, [-1, img_height, img_width, 1])
    h0 = BatchNormalization()(h0)
    h0 = ReLU()(h0)

    h1 = Concatenate(axis=3)([L, h0])
    h1 = Conv2D(128, (7,7), padding='same')(h1)
    h1 = BatchNormalization()(h1)
    h1 = ReLU()(h1)

    h2 = Concatenate(axis=3)([L, h1])
    h2 = Conv2D(64, (5,5), padding='same')(h2)
    h2 = BatchNormalization()(h2)
    h2 = ReLU()(h2)

    h3 = Concatenate(axis=3)([L, h2])
    h3 = Conv2D(64, (5,5), padding='same')(h3)
    h3 = BatchNormalization()(h3)
    h3 = ReLU()(h3)

    h4 = Concatenate(axis=3)([L, h3])
    h4 = Conv2D(64, (5,5), padding='same')(h4)
    h4 = BatchNormalization()(h4)
    h4 = ReLU()(h4)

    h5 = Concatenate(axis=3)([L, h4])
    h5 = Conv2D(32, (5,5), padding='same')(h5)
    h5 = BatchNormalization()(h5)
    h5 = ReLU()(h5)

    h6 = Concatenate(axis=3)([L, h5])
    h6 = Conv2D(2, (5,5), padding='same', activation='tanh')(h6)

    return tf.keras.Model(inputs=[L, z], outputs=h6)

def make_discriminator_model(img_height, img_width):
    '''
    Colorizer discriminator.

    Parameters
    ----------
    img_height : int
        Input image height.
    img_width : int
        Input image width.

    Returns
    -------
    discriminator: keras model
        An instance of discriminator.

    '''
    image = Input((img_height, img_width, 2), name='image')
    h0 = Conv2D(16, (5,5), padding='same')(image)
    h0 = LeakyReLU(0.2)(h0)

    h1 = Conv2D(32, (5,5), padding='same')(h0)
    h1 = BatchNormalization()(h1)
    h1 = LeakyReLU(0.2)(h1)

    h2 = Conv2D(64, (5,5), padding='same')(h1)
    h2 = BatchNormalization()(h2)
    h2 = LeakyReLU(0.2)(h2)

    h3 = Conv2D(128, (5,5), padding='same')(h2)
    h3 = BatchNormalization()(h3)
    h3 = LeakyReLU(0.2)(h3)

    h4 = Flatten()(h3)
    h4 = Dense(16)(h4)

    h5 = Dense(1)(h4)
    return tf.keras.Model(inputs=image, outputs=h5)

###############################################################################
# Loss
###############################################################################

def generator_loss(ab_real, ab_fake):
    '''
    MSE loss for generator. 

    Parameters
    ----------
    ab_real : tensor
        Real image `ab` color channels.
    ab_fake : tensor
        Fake image `ab` color channels from generator.

    Returns
    -------
    loss : tensor

    '''
    shape = tf.math.reduce_prod(ab_real.shape)
    return MSE(tf.reshape(ab_real, shape=(shape,)), tf.reshape(ab_fake, shape=(shape,)))

def discriminator_loss(logits_real, logits_fake):
    '''
    Binary cross entropy loss for discriminator.

    Parameters
    ----------
    logits_real : tensor
        Predicted logits for real image `ab` channels from the discriminator.
    logits_fake : tensor
        Predicted logits for generated image `ab` channels from the discriminator.

    Returns
    -------
    loss : tensor

    '''
    cross_entropy = BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(logits_real), logits_real)
    fake_loss = cross_entropy(tf.zeros_like(logits_fake), logits_fake)
    total_loss = real_loss + fake_loss
    return total_loss

###############################################################################
# Optimizer
###############################################################################

def generator_optimizer(lr):
    '''
    Helper function that returns an instance of Adam optimizer for generator.

    Parameters
    ----------
    lr : float
        Optimizer learning rate.

    Returns
    -------
    optimizer: Adam

    '''
    return Adam(learning_rate=lr)

def discriminator_optimizer(lr, clipvalue=None):
    '''
    Helper function that returns an instance of Adam optimizer for discriminator.

    Parameters
    ----------
    lr : float
        Optimizer learning rate.
    clipvalue : float, optional
        Value for gradient clipping. The default is None.

    Returns
    -------
    optimizer: Adam

    '''
    return Adam(learning_rate=lr, clipvalue=clipvalue)

