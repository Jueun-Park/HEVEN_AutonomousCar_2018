'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.

In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the uturn pictures index 0-999 in data/train/uturn
- put the uturn pictures index 1000-1400 in data/validation/uturn
- put the parking pictures index 12500-13499 in data/train/parking
- put the parking pictures index 13500-13900 in data/validation/parking
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:


```
data/
    train/
        uturn/
            uturn001.jpg
            uturn002.jpg
            ...
        parking/
            parking001.jpg
            parking002.jpg
            ...
    validation/
        uturn/
            uturn001.jpg
            uturn002.jpg
            ...
        parking/
            parking001.jpg
            parking002.jpg
            ...
```
'''

# 이 코드로 실험을 해볼까 합니다!


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Flatten, Dropout, Concatenate, Activation
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.applications.imagenet_utils import _obtain_input_shape


# dimensions of our images.
img_width, img_height = 32, 32

train_data_dir = 'data/TRAIN_DIR'
validation_data_dir = 'data/VALIDATION_DIR'
nb_train_samples = 7000
nb_validation_samples = 700
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

'''
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
'''

model = Sequential()
model.add(Conv2D(32, (1, 1), padding='valid',
                 input_shape=input_shape))
model.add(Activation('relu'))
#model.add(Conv2D(32, (2, 2), padding='same'))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (1, 1), padding='valid'))
model.add(Activation('relu'))
#model.add(Conv2D(64, (2, 2),padding='same'))
#model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7))
model.add(Activation('softmax'))


'''
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))
'''

sgd = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)





model.compile(loss='mean_squared_error',
              optimizer= sgd,
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('seven_class_squeeze.h5')