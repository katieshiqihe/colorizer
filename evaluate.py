#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 20:48:21 2021

@author: khe
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import model
import datautils
from train import IMG_HEIGHT, IMG_WIDTH, Z_DIM

# Get model instances
generator = model.make_generator_model(IMG_HEIGHT, IMG_WIDTH, Z_DIM)
discriminator = model.make_discriminator_model(IMG_HEIGHT, IMG_WIDTH)

# Load weigths
generator.load_weights('data/generator/weights')
discriminator.load_weights('data/discriminator/weights')

def predict_train(idx):
    '''
    Predict on training set.

    Parameters
    ----------
    idx : int
        Image index from training data.

    Returns
    -------
    fig : figure
        Matplotlib figure showing ground truth and generated images.
    generated_ab : numpy array
        Generator output. 

    '''
    train_data = datautils.load_training_data('data/youtube_data.h5', idx, idx+1, IMG_HEIGHT, IMG_WIDTH)
    L = train_data[:,:,:,:1]
    ab = train_data[:,:,:,1:]
    z = np.random.uniform(-1,1,(L.shape[0], Z_DIM))
    generated_ab = generator([L,z], training=False)
    lab_data = np.concatenate(((L[0]+1)*127.5, (generated_ab[0]+1)*127.5), axis=2).astype('uint8')
    img_fake = cv2.cvtColor(lab_data, cv2.COLOR_LAB2RGB)
    lab_data = np.concatenate(((L[0]+1)*127.5, (ab[0]+1)*127.5), axis=2).astype('uint8')
    img_real = cv2.cvtColor(lab_data, cv2.COLOR_LAB2RGB)   
    fig, ax = plt.subplots(1,2)
    ax[0].axes.set_axis_off()
    ax[1].axes.set_axis_off()
    ax[0].imshow(img_fake)
    ax[0].set_title('Predicted')
    ax[1].imshow(img_real)
    ax[1].set_title('Ground Truth')
    return fig, generated_ab
    
def predict_test():
    '''Generate colorized video on test data. 

    '''
    test_data = datautils.load_test_data('data/youtube_data.h5', img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    out = cv2.VideoWriter('data/test_color.mp4', cv2.VideoWriter_fourcc(*'avc1'), 30, (IMG_WIDTH, IMG_HEIGHT))
    for i in range(len(test_data)):
        L = test_data[i:i+1]
        z = np.random.uniform(-1,1,(L.shape[0], Z_DIM))
        generated_ab = generator([L,z], training=False)
        lab_data = np.concatenate(((L[0]+1)*127.5, (generated_ab[0]+1)*127.5), axis=2).astype('uint8')
        img = cv2.cvtColor(lab_data, cv2.COLOR_LAB2BGR)
        out.write(img)
    out.release()