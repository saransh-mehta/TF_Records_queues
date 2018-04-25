# image


You are building TWO models using Tensorflow to recognize apple fruit in images. 

1. You will NOT use pretrained models. You will do the training yourself. Create a training data set of atleast 100 images. Also create a validation data set of 10 images. Make sure the training dataset has images of different sizes and apples of different colors.
2. Convert the training data image to a tfrecord.
3. Create your OWN activation function (using tensorflow lambdas) instead of the normal ones. Justify why you are choosing this - if you choose others and this worked better, tell us about all the ones that didnt work !
4. Create two CNN models - one using your own activation function and the other using Relu. Create the models by training them on the training dataset.
5. Compare which one was better using the validation dataset.

Given task has been performed under following pipeline :-

# Dataset

The data has been collected manually from google image. 100 iimages of apples are downloaded and 94 images of other fruits/ vegetables
which are considered into non-apple class. 10 images from both apple, non-apple are separately downloaded for validation
The performance of model is expected to be not so good due to the irregularities in the data of non-apple class.
Below is the drive link for the created dataset.

https://drive.google.com/open?id=1I3_Gt-Wwv25wNsAPW27QZb3hwl1R1Yf5

# Renaming and format

The function rename_and_format() uses Opencv module to red images and save them in '.jpg' format with a serial number in their name.
This is done due to non-uniform image format in raw data.

# Data and Label Generation

Function create_data() creates numpy arrays having all images read in numpy array format and generates their corresponding labels
(not one-hot encoded now) returns imgList and LabelList

# TFRecord writing

Using the imgList and labelList, TFRecord has been saved separately for both train and validation.
TFRecord has following features, height, width, depth, label and raw image.

# Checking TFRecord

The saved TFRecords is loaded and we extract back the images so that to ensure that we have not saved the TFRecord incorrectly.

# Extracting everything in tensor format

While making the model, as tf requires everything to be in tensor format,
so the function extract_from_TFRecords_in_tensor_format() 
extracts the images from TFRecords and casts the features in tensors
It also makes batches of images with the given Batch Size.

# Model

the file tf_apple_recognizer_model.py contains the model
Architecture of the CNN is

conv2d with 32 filters
conv2d with 32 filters
maxPool with kernel_size (1, 2, 2, 1)
conv2d with 64 filters
conv2d with 64 filters
maxPool with kernel_size (1, 2, 2, 1)
conv2d with 128 filters
conv2d with 128 filters
maxPool with kernel_size (1, 2, 2, 1)

dense layer with 1024 units
dense layer with 512 units

finalOut dense layer with 2 units output

Softmax cross entropy loss has been used to classify along with AdamOptimizer

# Custom Activation Function

Two activation functions has been considered
First is tensorflow typical tf.nn.relu

Second is custom made Activation function as asked
I have choosen to go with making the output as absolute values. Because with images, the values are positive (pixel values)
So, better idea in activation could be to pull up the negative products (if there any) to the positive side 
with same magnitude rather than Zeroing them as relu does.

