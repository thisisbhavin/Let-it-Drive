import os
import cv2
import random
import numpy as np
from sklearn.utils import shuffle

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 68, 200, 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '/opt/carnd_p3/data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                
                center_image = img_preprocess(center_image)
                
                # Original centered image
                images.append(bgr2yuv(center_image))
                angles.append(center_angle)
                
                # image with gaussian noise added
                images.append(bgr2yuv(add_noise(center_image)))
                angles.append(center_angle)
                
                # brightened image
                images.append(bgr2yuv(add_brightness(center_image)))
                angles.append(center_angle)
                
                # flipped image
                flipped, flipped_angle = flip_img(center_image, center_angle)
                images.append(bgr2yuv(flipped))
                angles.append(flipped_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
            
def crop_img(img):
    """
    crops input image so that only road is visible
    """
    return img[65 : -25, :, :]

def resize_img(img):
    """
    resize input image 
    """
    return cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

def bgr2yuv(img):
    """
    converting fro BGR color format to YUV color space
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

def img_preprocess(img):
    """
    preprocessing on the input image
    first crops the image then resizes it
    """
    return resize_img(crop_img(img))

def flip_img(img, angle):
    """
    Given input image this function returns flipped image and angle multiplied by -1.0
    """
    return np.flip(img, 1), -1.0*angle if angle != 0 else 0

def add_brightness(img):
    """
    img: input image
    returns brightened or darkened image
    """
    scale = random.randrange(-100, 150, 50)
    scale = scale if scale !=0 else 100
        
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    channel_v = hsv[:, :, 2]
    
    if scale > 0:
        channel_v = np.where(channel_v <= 255 - scale, channel_v + scale, 255)
    else:
        channel_v = np.where(channel_v + scale <= 0, 0, channel_v + scale)
        
    hsv[:, :, 2] = channel_v
    
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr

def add_noise(img):
    """
    Adds random gaussian noise to input image
    """
    noisy_img = np.zeros(img.shape, np.uint8)
    mean = 0
    sigma = random.randrange(50, 500, 50)
    #print(mean, sigma)
    cv2.randn(noisy_img,mean,sigma)
    return cv2.add(img, noisy_img)
