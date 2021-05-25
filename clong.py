import csv
import cv2
import numpy as np
#import sklearn
from math import ceil
from sklearn.utils import shuffle

lines=[]
with open('./data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    
    num_samples=len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples=samples[offset:offset+batch_size]
            images =[]
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename=source_path.split('/')[-1]
                    current_path = './data2/IMG/' + filename
                    image =cv2.imread(current_path)
                    image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    measurement = float(batch_sample[3])
                    if i==0:
                        measurements.append(measurement)
                    elif i==1:
                        measurements.append(measurement+0.1)
                    else:
                        measurements.append(measurement-0.1)

            augmented_images, augmented_measurements =[],[]
            for image,measurement in zip(images,measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield shuffle(X_train, y_train)

batch_size=32

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
            
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
#model.fit(X_train, y_train, batch_size=128, validation_split=0.2, shuffle=True, nb_epoch = 6)

model.fit_generator(train_generator,
            steps_per_epoch=ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=ceil(len(validation_samples)/batch_size),
            epochs=5)

model.save('model.h5')
    