import math
import csv
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Conv2D, Lambda, MaxPooling2D, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import IMG_SHAPE, generator

def get_data():
    samples = []
    with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:
            samples.append(line)

    shuffle(samples)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples

def get_model():
    """
    Reference : http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    I have added dropout layers to reduce overfitting 
    """
    model = Sequential()
    
    # Normalization and zero centering.
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = IMG_SHAPE)) # 66x200
    
    model.add(Conv2D(24, 5, activation = 'relu', )) # 64x196x24
    model.add(MaxPooling2D(strides = (2, 2))) # 32x98x24
    model.add(Dropout(0.3))
    model.add(Conv2D(36, 5, activation = 'relu')) # 28x94x36
    model.add(MaxPooling2D(strides = (2, 2))) # 14x47x36
    model.add(Dropout(0.3))
    model.add(Conv2D(48, 5, activation = 'relu')) # 10x43x48
    model.add(MaxPooling2D(strides = (2, 2))) # 5x21x48
    model.add(Dropout(0.3))
    
    model.add(Conv2D(64, 3, activation = 'relu')) # 3x19x64
    model.add(Conv2D(64, 3, activation = 'relu')) # 1x17x64
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dense(100, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    
    model.add(Dense(1))
    
    model.summary()
    
    return model

def train_model(model, train, valid, batch_size):
    model.compile(loss='mse', optimizer='adam') # using adam optimizer
    
    # saving model at each epoch
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             mode='auto')
    
    model.fit_generator(generator(train, batch_size),
                        steps_per_epoch=math.ceil(len(train)/batch_size),             
                        validation_data=generator(valid, batch_size),
                        validation_steps=math.ceil(len(valid)/batch_size),
                        epochs=1,
                        verbose=1,
                        callbacks = [checkpoint])
    
    model.save('model.h5')
    
if __name__ == '__main__':
    batch_size = 128
    train, valid = get_data() # reads in rows from csv file and splits 80/20 into train and valid
    model = get_model() # create model
    train_model(model, train, valid, batch_size) # train model
