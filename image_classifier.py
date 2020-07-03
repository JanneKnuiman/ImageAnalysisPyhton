from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import argparse
import yaml

parser = argparse.ArgumentParser(description="Argument parser for image_classification")
parser.add_argument("-f", "--config_file", help="Configuration file to load.")
args = parser.parse_args()
yaml_file = args.config_file

with open(yaml_file) as file:
    settings = yaml.load(file, Loader=yaml.FullLoader)
    batch_size = settings['train']['batch_size']
    epochs = settings['train']['epochs']
    img_height = settings['target_size']['img_height']
    img_width = settings['target_size']['img_width']

def main():

    path = load_data()
    train_dir, validation_dir, total_train, total_eval = analyse_data(path)
    train_data_gen, val_data_gen = data_preparation(train_dir, validation_dir)
    model(total_train, total_eval, train_data_gen, val_data_gen)

"""
Function: This function downloads and unzips the data
Output: The path where the data is stored
"""
def load_data():

    url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

    try:
        path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=url, extract=True)
        path_unzipped = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
    except NameError:
        print("NameError: Make sure you use the right URL to extract the data from")

    return path_unzipped


"""
Input: The path to the directory where the data is stored
Function: This function constructs the paths to the train and evaluation directory, and gets the number of images in these directories 
Output: The training and validation directory (used for construction the ImageDataGenerator bjects) 
and the total number of training and evaluation images (used for the determination of the number of training and evaluation steps)
"""
def analyse_data(path):

    train_dir = os.path.join(path, 'train')
    validation_dir = os.path.join(path, 'validation')

    train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
    eval_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
    eval_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

    try:
        num_cats_train = len(os.listdir(train_cats_dir))
        num_dogs_train = len(os.listdir(train_dogs_dir))
        num_cats_eval = len(os.listdir(eval_cats_dir))
        num_dogs_eval = len(os.listdir(eval_dogs_dir))
    except FileNotFoundError as not_found:
        print('FileNotFoundError: The following directory is not found: ', not_found.filename)

    total_train = num_cats_train + num_dogs_train
    total_eval = num_cats_eval + num_dogs_eval

    """
    Possibility of printing information about the number of training and validation images per category, 
    useful for gaining insight in (the distribution of) the data
    """
    # print('total training cat images:', num_cats_train)
    # print('total training dog images:', num_dogs_train)
    #
    # print('total validation cat images:', num_cats_eval)
    # print('total validation dog images:', num_dogs_eval)
    # print("--")
    # print("Total training images:", total_train)
    # print("Total validation images:", total_eval)

    return train_dir, validation_dir, total_train, total_eval


"""
Input: Paths to the directories of the training and the validation data
Function: Constructing an ImageDataGenerator objects for the training and the validation.
While constructing the training ImageDataGenerator, some parameters are passed into 
the object to apply data augmentation on the training data.
The images are then loaded from the disk, and rescaling and resizing is applied (using the dimensions from train.yml
Output: The training and the validation ImageDataGenerator objects
"""
def data_preparation(train_dir, validation_dir):

    """
    Beneath is the definition of two generators for the training and the validation images,
    using the ImageDataGenerator class.

    This class reads the images from the disk and converts the images into a grid format as per their RGB content.
    It then  formats these images into pre-processed floating point tensors, which are eventually rescaled from values
    between 0 and 255 to values between 0 and 1, as neural networks prefer to deal with small input values.

    The training generator uses a few parameters, such as horizontal_flip and zoom_range.
    This parameters apply data augmentation on the training data, therefore increasing the variety in the training dataset.
    """
    train_image_generator = ImageDataGenerator(
                            rescale=1./255,
                            rotation_range=45,
                            width_shift_range=.15,
                            height_shift_range=.15,
                            horizontal_flip=True,
                            zoom_range=0.5
                            )
    validation_image_generator = ImageDataGenerator(rescale=1./255)


    # flow_from_directory method load images from the disk, applies rescaling, and resizes the images into the required dimensions
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=train_dir,
                                                               shuffle=True,
                                                               target_size=(img_height, img_width),
                                                               class_mode='binary')

    val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                  directory=validation_dir,
                                                                  target_size=(img_height, img_width),
                                                                  class_mode='binary')

    # Useful when testing: a set of 5 augmented imagese is extracted from the training generator and plotted with matplotlib.
    #augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    #plotImages(augmented_images)

    return train_data_gen, val_data_gen


"""
Input: An array of 5 images from the training generator
Function: This function plots the 5 images in the form of a grid with 1 row and 5 columns where the images are each placed in a column
"""
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


"""
Input: The total number of training and evaluation examples. The training and validation generator.
Function: This function constructs the model, by defining the layers and the dropout. 
It also compiles the model, using the right optimizer and loss function.
Then the model is trained, using the batch_size from the configuration yaml file, and the number of training and evalution examples.
Eventually, the dictonairy aquired during training is used to call a function that plots the accuracy and loss during training.
Output:
"""
def model(total_train, total_eval, train_data_gen, val_data_gen):

    start_load_model = time.time()
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width ,3)),
        MaxPooling2D(),
        Dropout(0.1),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.1),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1)
    ])

    print(f'Model loaded in ', {time.time()-start_load_model}, 'seconds')

    # Comilation of the model. Accuracy is printed during training for each epoch
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # View all the layers of the Sequential network
    model.summary()

    # Training of the model
    start_train_model = time.time()
    history = model.fit(
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_eval // batch_size
    )
    time_train = time.time()-start_train_model
    print(f'Model trained in ', time_train/60, 'minutes')

    try:
        train_acc = history.history['accuracy']
        eval_acc = history.history['val_accuracy']
        train_loss = history.history['loss']
        eval_loss = history.history['val_loss']
        plot_training(train_acc, eval_acc, train_loss, eval_loss)
    except KeyError as e:
        print('KeyError: The following key does not exists:', e.args[0])


"""
Input: Accuracy and loss of training and evaluation
Function: This function plots the training and evaluation loss and accuracy over the range of epochs.
"""
def plot_training(train_acc, eval_acc, train_loss, eval_loss):
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc, label='Training Accuracy')
    plt.plot(epochs_range, eval_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, eval_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

main()