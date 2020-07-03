# Image Analysis pg6
Image Analysis - project group 6: Valerie Verhalle, Sanne Schroduer &amp; Janne Knuiman

Start date: 02-03-2020

This project is set up to classify images using a Sequential sklearn model into a category: cat or dog.
The input data is downloaded from: https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip


## Installation of the required modules
To install all the models that are required for the deployment of the classifier, use pip to install the modules in requirementx.txt.

```pip install -r requirements.txt```

## Installation of CUDA
To run the script on GPU the image classifier is tested with CUDA. Used graphics card: NVIDIA GeForce GTX 1050.
Requirements install CUDA: Microsoft Windows 10, Microsoft Visual Studio and, NVIDIA CUDA Toolkit (available at http://developer.nvidia.com/cuda-downloads)
Make sure the NVIDIA software packages maches the versions of Tensorflow: Tested with CUDA version 10.1 for Tensorflow version 2.1.0.

## Usage of the classifier
The image classifier can be called using the command line.
The training variables (like batch-size) can be specified in a configuration .yml file, 
and are loaded in the python classifier.
Using the ```-f``` or ```--config_file``` argument, you can specify the path to your configuration yaml file:

```python image_classifier.py -f [path\to\your\.yml]```

