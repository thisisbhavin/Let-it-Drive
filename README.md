# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
 * Build, a convolution neural network in Keras that predicts steering angles from images
 * Train and validate the model with a training and validation set
 * Test that the model successfully drives around track one without leaving the road
 * Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/angle_distribution.PNG "Angles distribution"
[image2]: ./images/flip.PNG "Flipped"
[image3]: ./images/brightness.PNG "Brightened Image"
[image4]: ./images/noise.PNG "noisy Image"
[image5]: ./images/input_image.PNG "input Image"
[image6]: ./images/preprocess.PNG "preprocess"

## Rubric Points
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
 * model.py containing the script to create and train the model
 * utils.py contains utility functions for preprocessing, augmenting data such as flip, brightness, noise
 * drive.py for driving the car in autonomous mode
 * model.h5 containing a trained convolution neural network 
 * video.mp4 video of driving simulation
 * writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
I had to tweak the code a little bit to accomodate for the preprocessing done at the training time. it can be found on line 65 in drive.py

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I referred NVIDIA's paper on self driving car architecture that has tried and tested. I have slightly modified the network presented in the paper ad added dropout layers to reduce overfitting.

the network architecture is from line 27 to 52 in model.py file it uses 5x5 and 3x3 convolutions with number of filters ranging from 24 to 64

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (line 30 model.py). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting 

The model was trained and validated on different data sets to ensure that the model was not overfitting and  also more the data the better the model so I have created few data augmentation function like flip, add_noise, add_brightness (utils.py). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I used emperically proved network that was built for exactly this task and it was from NVIDIA, so I need not had to woory about trying out many network before settling on one.

though I want to mention that original network in the paper did not contain dropout layer, I added them to combat overfitting.

Using this model I got very good accuracy.

The final step was to run the simulator to see how well the car was driving around track one. It was working perfectly

#### 2. Final Model Architecture

The final model architecture (model.py lines 27-52) consisted of a convolution neural network with the following layers and layer sizes:

|Layer (type)                | Output Shape         |     Param #   |
|----------------------------|----------------------|---------------|
|lambda_2 (Lambda)           | (None, 68, 200, 3)   |     0         |
|conv2d_2 (Conv2D)           | (None, 64, 196, 24)  |     1824      |
|max_pooling2d_2 (MaxPooling2| (None, 32, 98, 24)   |     0         |
|dropout_1 (Dropout)         | (None, 32, 98, 24)   |     0         |
|conv2d_3 (Conv2D)           | (None, 28, 94, 36)   |     21636     |
|max_pooling2d_3 (MaxPooling2| (None, 14, 47, 36)   |     0         |
|dropout_2 (Dropout)         | (None, 14, 47, 36)   |     0         |
|conv2d_4 (Conv2D)           | (None, 10, 43, 48)   |     43248     |
|max_pooling2d_4 (MaxPooling2| (None, 5, 21, 48)    |     0         |
|dropout_3 (Dropout)         | (None, 5, 21, 48)    |     0         |
|conv2d_5 (Conv2D)           | (None, 3, 19, 64)    |     27712     |
|conv2d_6 (Conv2D)           | (None, 1, 17, 64)    |     36928     |
|dropout_4 (Dropout)         | (None, 1, 17, 64)    |     0         |
|flatten_1 (Flatten)         | (None, 1088)         |     0         |
|dense_1 (Dense)             | (None, 100)          |     108900    |
|dropout_5 (Dropout)         | (None, 100)          |     0         |
|dense_2 (Dense)             | (None, 50)           |     5050      |
|dense_3 (Dense)             | (None, 10)           |     510       |
|dense_4 (Dense)             | (None, 1)            |     11        |
|____________________________|______________________|_______________|
|Total params:               | 245,819              |               |
|Trainable params:           | 245,819              |               |
|Non-trainable params:       | 0                    |               |
|____________________________|______________________|_______________|

#### 3. Creation of the Training Set & Training Process

I started with the data provided in the workspace. 
First I looked at the distibution of angles and half of the images were close to 0 degrees.

![alt text][image1]

the data contained total of 8036 center images. After 80/20 split it becomes 6428 training and 1608 validaton data points. Assuming this to be very small number of data I decided to augment dataset.

I created 3 functions:
 * flip_img()
 * add_brightness()
 * add_noise()

__flip_img()__ - given input image it flips right to left and also multiplies -1.0 with angle corresponding to the image.
 ```python
 def flip_img(img, angle):
    """
    Given input image this function returns flipped image and angle multiplied by -1.0
    """
    return np.flip(img, 1), -1.0*angle if angle != 0 else 0
```    
![alt text][image2]

---
__add_brightness()__ - scales v channel from HSV by random amount
```python
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
```
![alt text][image3]

---
__add_noise()__ - Adds gaussian noise to image.
```python
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
```
![alt text][image4]

---
These augmentations were applied to each images, this makes __trainig set to 6428 * 4 = 25712 images__ and __validation set to 1608 * 4 = 6432 images__

Before starting with the training I had to crop and resize the image (66x200x3)  since image contained unncessary background info and car hood. all the input image to network are transformed into YUV space

__Input image__

![alt text][image5]

__Cropped and Resized image__

![alt text][image6]

Now using this data, which is being generated on the fly while training the network.

I have used batch size of 128 but effective batch size is 4*128 since for each batch 4 augmentaed images are generated. i trained network for 20 epochs and got around 0.0091 loss for train and valid set.
