#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for generating, loading, and processing training and test images.

@author: khe
"""
from pytube import YouTube
import cv2
import numpy as np
import tables
import os

###############################################################################
# Data Generation
###############################################################################

class TrainTable(tables.IsDescription):
    """Table in hdf5 file for storing color image array.
    
    """
    L = tables.UInt8Col(pos=0)
    A = tables.UInt8Col(pos=1)
    B = tables.UInt8Col(pos=2)
    
class TestTable(tables.IsDescription):
    """Table in hdf5 file for storing grayscale image array.
    
    """
    L = tables.UInt8Col(pos=0)
    
def capture_youtube(url, filename, skip_open=0, skip_end=0, interval=None, mode='LAB'):
    """
    Download a YouTube video, take screenshots, and return them as array.

    Parameters
    ----------
    url : str
        Link to Youtube video.
    filename : str
        Output filename without extension.
    skip_open : int, optional
        Number of seconds to skip at the beginning of video. The default is 0.
    skip_end : int, optional
        Number of seconds to skip at the end of video. The default is 0.
    interval : int, optional
        Number of frames between each screenshot, default is number of frames 
        per second.
    mode: str, optional
        `RGB`, `LAB` (color) or `L` (grayscale). The default is `LAB`. 

    Returns
    -------
    None.

    """
    assert mode in ('RGB', 'L', 'LAB')
    
    # Download video from YouTube
    video = YouTube(url)
    
    for stream in video.streams.filter(file_extension = "mp4"):
        # Get the 360p video stream
        if stream.mime_type == 'video/mp4':
            if stream.resolution == '360p':
                if not os.path.isfile(filename+'.mp4'):
                    stream.download(filename=filename, output_path='data')
                break
            
    # Extract frames from video
    vid_cap = cv2.VideoCapture(filename+'.mp4')
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    total_frames = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if interval is None:
        interval = int(fps)
    
    frame_count = 0
    if mode in ('RGB', 'LAB'):
        data = np.empty((0, 360, 480, 3)).astype('uint8')
    else:
        data = np.empty((0, 360, 480, 1)).astype('uint8')
        
    while vid_cap.isOpened():
        
        success,image = vid_cap.read() 
        
        if not success:
            break
        
        # Skip the openning and title (first 75 seconds)
        if frame_count >= fps*skip_open:
            # Get one image per second of footage
            # The videos are about 50-60 minutes long, this will result in about 3000-3600 images per video
            if frame_count % interval == 0:
                # Trim if aspect ratio is not right
                if image.shape[1] == 640:
                    image = image[:,80:-80,:]
                # Default color scheme in openCV is BGR, covert to RGB
                if mode == 'RGB':
                    # Skip frame if it's black screen
                    if np.nanmax(image) < 5:
                        continue
                    image = image[:,:,::-1]
                elif mode == 'LAB':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
                    # Skip frame if it's black screen
                    if np.nanmax(image[:,:,0]) < 5:
                        continue
                else: 
                    # Skip frame if it's black screen
                    if np.nanmax(image[:,:,0]) < 5:
                        continue                    
                    image = image[:,:,:1]
                data = np.concatenate((data, np.expand_dims(image, axis=0)))
            
        frame_count += 1
        
        # Skip the end
        if frame_count >= (total_frames-fps*skip_end):
            break
    
    vid_cap.release()
    cv2.destroyAllWindows()
    
    return data

def build_database():
    '''Create images from YouTube videos and store into HDF5 file.

    '''

    # Create HDF5 file
    mdb = tables.open_file('data/youtube_data.h5', mode="w")
    filters = tables.Filters(complevel=5, complib='blosc')
    # Create tables
    mdb.create_table('/', 'Train', TrainTable, filters=filters)
    mdb.create_table('/', 'Test', TestTable, filters=filters)
    mdb.flush()
    mdb.close()
    
    # Training Set
    video_urls = [
        'https://www.youtube.com/watch?v=aRRYIe6hXTQ&list=PLklyfwlKNjxD52EQbChCopHxWdAXBX0cg',
        'https://www.youtube.com/watch?v=NZlBM8hw3cg&list=PLklyfwlKNjxD52EQbChCopHxWdAXBX0cg&index=3',
        'https://www.youtube.com/watch?v=_PhHKIufB4Q&list=PLklyfwlKNjxD52EQbChCopHxWdAXBX0cg&index=4',
        'https://www.youtube.com/watch?v=w4Jfm4J-9tw',
        'https://www.youtube.com/watch?v=PmZP_efIOhQ'
        ]
    
    for i in range(len(video_urls)):
        filename = 'video_%s'%i
        data = capture_youtube(video_urls[i], filename, 75, 60)
        L = data[:,:,:,0].flatten()
        A = data[:,:,:,1].flatten()
        B = data[:,:,:,2].flatten()
        
        mdb = tables.open_file('data/youtube_data.h5', mode="r+")
        data_table = mdb.root.Train
        data_table.append(np.stack((L,A,B), axis=1))
        data_table.flush()
        mdb.close()
        
    # Test Set
    data = capture_youtube('https://www.youtube.com/watch?v=QSegeI5Qn6A',
                           'data/test_data', interval=1, mode='L')
    L = data[:,:,:,0].flatten()
    
    mdb = tables.open_file('data/youtube_data.h5', mode="r+")
    data_table = mdb.root.Test
    data_table.append(L)
    data_table.flush()
    mdb.close()
    
###############################################################################
# Image Loading and Preprocessing
###############################################################################
    
def trim_images(image, height, width):
    '''
    Trim images to specified height and width from the center.

    Parameters
    ----------
    image : numpy array
        An array of images, expecting 4 dimensions in the format of `NHWC`.
    height : int
        Output image height.
    width : int
        Output image width.

    Returns
    -------
    out : numpy array
        An array of output images.

    '''
    image_height = image.shape[1]
    image_width = image.shape[2]
    
    out = image.copy()
    if image_height > height:
        top = (image_height-height)//2
        bottom = image_height-height-top
        out = out[:,top:-bottom, :, :]
    if image_width > width:
        left = (image_width-width)//2
        right = image_width-width-left
        out = out[:, :, left:-right, :]
    return out

def load_training_data(filename, start_idx=None, end_idx=None, img_height=360, img_width=480):
    '''
    Get a set of training images from HDF5.

    Parameters
    ----------
    filename : str
        Full path to HDF5.
    start_idx : int, optional
        Starting image index, default is None (load all images).
    end_idx : int, optional
        Ending image index, default is None (load all images).
    img_height : int, optional
        Output image height, default is None (no trimming in height).
    img_width : int, optional
        Output image width, default is None (no trimming in width).

    Returns
    -------
    data : numpy array
        Image array in float, ranges between -1 and 1 in `LAB` colorspace and 
        `NHWC` format.

    '''
    mdb = tables.open_file(filename)
    tbl = mdb.root.Train
    if (start_idx is None) and (end_idx is None):
        L = tbl.cols.L[:]
        A = tbl.cols.A[:]
        B = tbl.cols.B[:]
    else:
        assert (start_idx is not None) and (end_idx is not None), 'Please provide both start and end indices.'
        start_idx = int(start_idx*360*480)
        end_idx = int(end_idx*360*480)
        L = tbl.cols.L[start_idx:end_idx]
        A = tbl.cols.A[start_idx:end_idx]
        B = tbl.cols.B[start_idx:end_idx]
    mdb.close()
    
    img_size = (360, 480)
    n = int(L.shape[0]/np.prod(img_size))
    L = L.reshape(n, img_size[0], img_size[1])
    A = A.reshape(n, img_size[0], img_size[1])
    B = B.reshape(n, img_size[0], img_size[1])
    
    data = np.stack((L, A, B), axis=3)/127.5-1
    data = np.array([cv2.resize(x, (img_width, img_height)) for x in data])
    return data

def load_test_data(filename, start_idx=None, end_idx=None, img_height=None, img_width=None):
    '''
    Get a set of test images from HDF5.

    Parameters
    ----------
    filename : str
        Full path to HDF5.
    start_idx : int, optional
        Starting image index, default is None (load all images).
    end_idx : int, optional
        Ending image index, default is None (load all images).
    img_height : int, optional
        Output image height, default is None (no trimming in height).
    img_width : int, optional
        Output image width, default is None (no trimming in width).

    Returns
    -------
    data : numpy array
        Image array in float, ranges between -1 and 1 in `LAB` colorspace and 
        `NHWC` format.

    '''
    mdb = tables.open_file(filename)
    tbl = mdb.root.Test
    if (start_idx is None) and (end_idx is None):
        L = tbl.cols.L[:]
    else:
        assert (start_idx is not None) and (end_idx is not None), 'Please provide both start and end indices.'
        start_idx = int(start_idx*360*480)
        end_idx = int(end_idx*360*480)
        L = tbl.cols.L[start_idx:end_idx]
    mdb.close()
    
    img_size = (360, 480)
    n = int(L.shape[0]/np.prod(img_size))
    L = L.reshape(n, img_size[0], img_size[1])
    
    data = np.expand_dims(L, axis=3)/127.5-1
    data = np.expand_dims(np.array([cv2.resize(x, (img_width, img_height)) for x in data]), axis=3)
    return data