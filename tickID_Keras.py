#Author: Ali Khalighifar
#Date: 03/21/2022

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing import image
from keras.models import load_model

''' Here is the folder design: We will have a base directory that has all 
tick images. Then we will randomly copy images to train and test directory,
using shutil.copy(). Next, we will do all the training and evaluation
process on those two directories, and never touch the main/base directory.'''

base_dir = 'Path to Base Directory'
train_dir = 'Path to Train Directory'
test_dir = 'Path to Test Directory'
num_classes = 'Number of classes/tick species' #Alternatively, you can do the other method below

#We will save model checkpoints in this folder to save the best model
checkpoint_dir = 'Path to Checkpoint Directory'

'''Rescaling images, and randomly choosing 20% of images in the Train set to be used for the
validation process. We may also want to change the batch size depending 
on the number of images per directory. But whatever number we choose, it's
better to be same for both directories (i.e., Train and Validation sets).
Note: Validation set is different from Test set.'''
image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2)
train_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory=train_dir,
                                                 shuffle=True,
                                                 target_size=(299, 299), 
                                                 subset="training",
                                                 class_mode='categorical')

validation_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory=train_dir,
                                                 shuffle=True,
                                                 target_size=(299, 299), 
                                                 subset="validation",
                                                 class_mode='categorical')

#num_classes = train_dataset.class_names #That's the alternative method for # of classes

#Using Keras function to augment images
data_augmentation = keras.Sequential(
    [keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
   keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ]
)

#Loading the model without the classification layers, 
#and freezing convolutional base
base_model = keras.applications.inception_v3.InceptionV3(
    weights='imagenet',  
    input_shape=(299, 299, 3),
    include_top=False)
base_model.trainable = False

#Setting up the model configuration

initial_epochs = 20
inputs = keras.Input(shape=(299, 299, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.inception_v3.preprocess_input(x)
x = base_model(x, training=False)
#Add a Global average layer
x = keras.layers.GlobalAveragePooling2D()(x)
#Add a dropout layer to avoid overfitting
x = keras.layers.Dropout(0.2)(x)
predictions = keras.layers.Dense(3, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam', 
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=keras.metrics.Accuracy())
history = model.fit(train_dataset, 
                    epochs=initial_epochs, 
                    validation_data=validation_dataset)

#Here we start fine-tuning the model we just trained above
fine_tune_epochs = 20
total_epochs =  initial_epochs + fine_tune_epochs
#Un-freezing the convolutional base to fine-tune the model
base_model.trainable = True
#Choose a name for your model to save
MODEL_FILE = 'filename.model'
model.compile(optimizer=keras.optimizers.Adam(1e-3),  
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=keras.metrics.Accuracy())
#Check to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

#Fine-tune from this layer onwards to reduce computational cost
fine_tune_at = 100

#Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

#If the performance doesn't improve, stop the training after 5 epochs
earlyStopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='max')
#Save the model checkpoints to choose the best model
model_checkpoint = ModelCheckpoint(filepath=checkpoint_dir,
                                   monitor='val_accuracy',
                                   mode='max',
                                   save_best_only=True)
#Reduce the learning rate gradually by 0.1 and monitor accuracy
reduce_lr_loss = ReduceLROnPlateau(monitor='val_accuracy',
                                   mode='max',
                                   factor=0.1,
                                   patience=5, verbose=1)

callbacks = [earlyStopping, model_checkpoint, reduce_lr_loss]

#Re-configure the model to fine tune
history_fine = model.fit(train_dataset, epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset,
                         callbacks = callbacks)
model.save(MODEL_FILE)

#Use this function to predict on images in Test set
def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    preds = model.predict(x)
    return preds[0]

'''Below is an example code to test on one image. However, you can make a
for loop to go through all images in the Test directory, and apply the 
predict function. You can also save the results of prediction as a text file.'''
img = image.load_img('one image from Test Directory.png', target_size=(299, 299))
preds = predict(load_model(MODEL_FILE), img)
