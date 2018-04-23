'''
Here we will load the images, rename all and combine apple 
and non-apple with respective labels and shuffle
and split them in train, test and validation them together and write it into TFrecords for training
'''
import numpy as np
import os
import cv2

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf

DATA_PATH = os.path.abspath('../own_raw_data')
TF_RECORD_PATH = os.getcwd()
#CLASSES = os.listdir(DATA_PATH)
CLASSES = ['apples', 'non_apples']
SEED = 2
IMAGE_SIZE = (250, 250)
np.random.seed(SEED)

'''
def rename_and_format():

	applesList = os.listdir(os.path.join(DATA_PATH, CLASSES[0]))
	nonApplesList = os.listdir(os.path.join(DATA_PATH, CLASSES[1]))

	fileIndex = 1
	for file in applesList:
		img = cv2.imread(os.path.join(DATA_PATH, CLASSES[0], file))
		cv2.imwrite(os.path.join(DATA_PATH, 'apples_renamed', (str(fileIndex) + '.jpg')), img)
		fileIndex += 1
	# we will add '_' after serial so as to distinguish it from apple files
	fileIndex = 1
	for file in nonApplesList:
		img = cv2.imread(os.path.join(DATA_PATH, CLASSES[1], file))
		cv2.imwrite(os.path.join(DATA_PATH, 'non_apples_renamed', (str(fileIndex) + '_.jpg')),
			img)
		fileIndex += 1
	print('renamed files successfully and saved')
'''

def create_data():
	imgList = []
	labelList = []   #use 0 for apples, 1 for non_apples

	# for apples 
	files = os.listdir(os.path.join(DATA_PATH, CLASSES[0]))
	for file in files:
		img = cv2.imread(os.path.join(DATA_PATH, CLASSES[0], file)) # this creates a numpy array for image
		img = cv2.resize(img, IMAGE_SIZE)
		imgList.append(img)
		labelList.append(0)

	files = os.listdir(os.path.join(DATA_PATH, CLASSES[1]))
	for file in files:
		img = cv2.imread(os.path.join(DATA_PATH, CLASSES[1], file)) # this creates a numpy array for image
		img = cv2.resize(img, IMAGE_SIZE)
		imgList.append(img)
		labelList.append(1)

	imgList, labelList = shuffle(imgList, labelList, random_state = SEED)
	imgList = np.array(imgList)
	labelList = np.array(labelList)
	print(" Images and labels ready after shuffling")
	print('imgList shape = ', imgList.shape)
	return imgList, labelList


def write_TFrecord(data, label, writePath, name):
	# creating filename and writer object
	filename = os.path.join(writePath, name + '.tfrecords')
	writer = tf.python_io.TFRecordWriter(filename)

	for i in range(len(data)):
		# as tfRecords save features as serialized strings,
		# we will convert it to strings
		image = data[i].tostring()
		featureDicti = {'height' : tf.train.Feature(int64_list = tf.train.Int64List(value = [data[i].shape[0]])),
		'width' : tf.train.Feature(int64_list = tf.train.Int64List(value = [data[i].shape[1]])),
		'depth' : tf.train.Feature(int64_list = tf.train.Int64List(value = [data[i].shape[2]])),
		'label' : tf.train.Feature(int64_list = tf.train.Int64List(value = [label[i]])),
		'image' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])) }
		# example is the structure that stores features of tfRecords
		example = tf.train.Example(features = tf.train.Features(feature = featureDicti))
		writer.write(example.SerializeToString())
	writer.close()
	print("TFrecord wriiten : ", filename)

# here we will also create a function which will test that we have correctly 
# saved our TFRecord by extracting the image from it
def check_TFRecord(fileName):
	#creating TFRecord reader object
	filePath = os.path.abspath(fileName + '.tfrecords')
	recordReader = tf.python_io.tf_record_iterator(filePath)
	extractedImages = []
	for record in recordReader:
		# creating object of example structure
		example = tf.train.Example()
		# since in records, we stored in form of serialized strings
		example.ParseFromString(record)
		height = int(example.features.feature['height'].int64_list.value[0])
		width = int(example.features.feature['width'].int64_list.value[0])
		depth = int(example.features.feature['depth'].int64_list.value[0])
		label = int(example.features.feature['label'].int64_list.value[0])
		image = example.features.feature['image'].bytes_list.value[0]

		imgReshaped = np.fromstring(image, dtype = np.uint8).reshape((height, width, depth))
		extractedImages.append(imgReshaped)

	return extractedImages

def plot_image_from_list(imgList, imgNum):

	cv2.imshow('reconstructed_image', imgList[imgNum])
	cv2.waitKey(0)

'''
Here we will also make a function which will extract from TFRecords and make everthing
in suitable tensor formats so that we can later directly use this function to supply
to the queue while training
'''
def extract_from_TFRecords_in_tensor_format(queueName, batchSize):

	#making reader object
	reader = tf.TFRecordReader()
	_, serializedExample = reader.read(queueName)

	featureDicti2 = {
	'height': tf.FixedLenFeature([], tf.int64),
	'width': tf.FixedLenFeature([], tf.int64),
	'depth' : tf.FixedLenFeature([], tf.int64),
	'label' : tf.FixedLenFeature([], tf.int64),
	'image': tf.FixedLenFeature([], tf.string),
	}

	features = tf.parse_single_example(serializedExample, features = featureDicti2)

	height = tf.cast(features['height'], tf.int32)
	width = tf.cast(features['width'], tf.int32)
	depth = tf.cast(features['depth'], tf.int32)
	label = tf.cast(features['label'], tf.int32)

	image = tf.decode_raw(features['image'], tf.uint8) # decoding string in int format
	image = tf.reshape(image, tf.stack([height, width, depth]))

	imageSizeTensor = tf.constant((IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=tf.int32)

	#we need an extra reshape below because shuffle_batch requires each internal dimension
	# which it gets by reading tensor
	image = tf.reshape(image, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])


	images, labels = tf.train.shuffle_batch( [image, label],batch_size=batchSize, capacity=30,
		num_threads=2, min_after_dequeue=10)
	print('extracted from TFRecord in required tensor format for model')

	return images, labels


imagesList, labelsList = create_data()
# Now we will split train and test set and store 2 separate TFrecord for train and test
# we need 10 test images
xTrain, xTest, yTrain, yTest = train_test_split(imagesList, labelsList, test_size = 0.1)
write_TFrecord(xTrain, yTrain, TF_RECORD_PATH, 'train_apple_recognizer')
write_TFrecord(xTest, yTest, TF_RECORD_PATH, 'test_apple_recognizer')
imgList = check_TFRecord('train_apple_recognizer')
#plot_image_from_list(imgList, 2)

'''
filenameQueue = tf.train.string_input_producer(
    ['train_apple_recognizer.tfrecords'], num_epochs=10) #num_epoch times file will be read
images, labels = extract_from_TFRecords_in_tensor_format(filenameQueue, 2)
'''
