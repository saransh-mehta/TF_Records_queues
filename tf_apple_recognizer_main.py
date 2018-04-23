'''
Here we will make our model CNN
'''

import tensorflow as tf
import numpy as np
import os

from tf_apple_recognizer_raw_pre import extract_from_TFRecords_in_tensor_format
from tf_apple_recognizer_model import create_model

TF_RECORD_TRAIN_PATH = os.path.abspath('train_apple_recognizer.tfrecords')
TF_RECORD_VAL_PATH = os.path.abspath('validation_apple_recognizer.tfrecords')
LOG_DIR = os.path.join(os.getcwd(), 'tmp')

THREADS_NUM = 1
NUM_CLASSES = 2
BATCH_SIZE = 3
DROP_OUT = 0.5
EPOCHS = 1000

filenameQueue = tf.train.string_input_producer(
    [TF_RECORD_TRAIN_PATH], num_epochs = 20) #num_epoch times file will be read

filenameQueueVal = tf.train.string_input_producer(
    [TF_RECORD_VAL_PATH], num_epochs = 20) #num_epoch times file will be read

images, labels = extract_from_TFRecords_in_tensor_format(filenameQueue, BATCH_SIZE)

imagesVal, labelsVal = extract_from_TFRecords_in_tensor_format(filenameQueueVal, BATCH_SIZE)

create_model(images, labels, NUM_CLASSES, activation = 'relu')

initGlobal = tf.global_variables_initializer()
initLocal = tf.local_variables_initializer()

with tf.Session() as sess:

	sess.run(initGlobal)
	sess.run(initLocal)

	trainWriter = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
	testWriter = tf.summary.FileWriter(LOG_DIR + '/test', sess.graph)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	for i in range(194 *4):

		#batchX, batchY = data.get_next_batch(BATCH_SIZE, trainX, trainY)
		sess.run(train, feed_dict = {keep_prob : DROP_OUT})

		if i % 10 == 0:
			# calculating train accuracy
			acc, lossTmp = sess.run([accuracy, loss], feed_dict = {keep_prob : DROP_OUT})
			print('Iter: '+str(i)+' Minibatch_Loss: '+"{:.6f}".format(lossTmp)+' Train_acc: '+"{:.5f}".format(acc))
	coord.request_stop()
	coord.join(threads)
	'''			
	for i in range(5):
		# calculating test accuracy
		testBatchX, testBatchY = data.get_next_batch(BATCH_SIZE, testX, testY)
		testAccuracy = sess.run(accuracy, feed_dict = {x : testBatchX, y : testBatchY, keep_prob : DROP_OUT})
		print('test accuracy : ', testAccuracy)
	'''