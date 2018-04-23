'''
Here we will make our model CNN
'''
with tf.name_scope('placeholders') as scope:
	x = tf.placeholder(shape = [BATCH_SIZE, 48, 48, 1], name = 'input', dtype = tf.float32)
	y = tf.placeholder(shape = [BATCH_SIZE, CLASS_NUM], name = 'labels', dtype = tf.float32)
with tf.name_scope('cnn') as scope:
	# now here is an issue with tensorflow, In the convolution filter, we can't directly pass
	# a list like [3, 3, 1, 32] because it considers it a list which has rank 1, but it requires rank 4
	#input, hence we need to create a variable first of the required shape
	def create_filter(shape):
		filters = tf.Variable(tf.truncated_normal(shape, stddev = 0.1))
		return filters

	conv1 = tf.nn.relu(tf.nn.conv2d(x, filter = create_filter([3, 3, 1, 32]), strides = [1, 1, 1, 1],
	 padding = 'SAME', name = 'conv1'))
	# here we have defined our first layer with a convolution window of 3x3 and 32 feature maps
	# the 1 in between shows that initially we are having only 1 feature map (tht is the mono channel of image)

	conv2 = tf.nn.relu(tf.nn.conv2d(conv1, filter = create_filter([3, 3, 32, 32]), strides = [1, 1, 1, 1],
	 padding = 'SAME', name = 'conv2'))

	maxPool1 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME',
		name = 'maxPool1')
	# after this the image size will be reduced to 24x24x32
	conv3 = tf.nn.relu(tf.nn.conv2d(maxPool1, filter = create_filter([3, 3, 32, 64]), strides = [1, 1, 1, 1],
	 padding = 'SAME', name = 'conv3'))

	conv4 = tf.nn.relu(tf.nn.conv2d(conv3, filter = create_filter([3, 3, 64, 64]), strides = [1, 1, 1, 1],
	 padding = 'SAME', name = 'conv4'))

	maxPool2 = tf.nn.max_pool(conv4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME',
		name = 'maxPool2')
	# now the image size has reduced to 12x12x64 after this maxpooling
	conv5 = tf.nn.relu(tf.nn.conv2d(maxPool2, filter = create_filter([3, 3, 64, 128]), strides = [1, 1, 1, 1],
	 padding = 'SAME', name = 'conv5'))

	conv6 = tf.nn.relu(tf.nn.conv2d(conv5, filter = create_filter([3, 3, 128, 128]), strides = [1, 1, 1, 1],
	 padding = 'SAME', name = 'conv6'))

	maxPool3 = tf.nn.max_pool(conv6, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME',
		name = 'maxPool3')
	# now the image has reduced to 6x6x128 after this maxpooling
	# here we can see that we have reduced the height, width dimensions of the image, but increased 
	# the number of features of the image, hence making a balance between the number of neurons.
	flatten = tf.reshape(maxPool3, (BATCH_SIZE, 6 * 6 * 128))
	# here we have unrolled the whole structure of image into one dimensional tensors so that
	# we can connect it to dense layers
with tf.name_scope('dense') as scope:
	dense1 = tf.nn.relu(tf.layers.dense(flatten, units = 1024, name = 'dense1'))
	# thus here the neuron count in our model is 1024*6*6*128
	keep_prob = tf.placeholder(tf.float32)
	# the dropout ratio has to be a placeholder which will be fed value at training like x
	dropOut1 = tf.nn.dropout(dense1, keep_prob = keep_prob, name = 'drop1')
	# probability that an element is kept is keep_prob
	dense2 = tf.nn.relu(tf.layers.dense(dropOut1, units = 512, name = 'dense2'))
	dropOut2 = tf.nn.dropout(dense2, keep_prob = keep_prob, name = 'drop2')
with tf.name_scope('out_layer') as scope:
	finalOutput = tf.nn.softmax(tf.layers.dense(dropOut2, units = 7, name = 'output'))
	# here in the final layer we have choosen softmax as activation because we need classification and
	# softmax helps in rounding up the probability into a certain class
with tf.name_scope('train') as scope:
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = finalOutput, labels = y))
	optimizer = tf.train.AdamOptimizer()
	train = optimizer.minimize(loss)

with tf.name_scope('accuracy') as accuracy:
	correctPrediction = tf.equal(tf.argmax(finalOutput, axis = 1), tf.argmax(y, axis = 1))
	accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32)) * 100

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	trainWriter = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
	testWriter = tf.summary.FileWriter(LOG_DIR + '/test', sess.graph)

	for i in range(EPOCHS):
		batchX, batchY = data.get_next_batch(BATCH_SIZE, trainX, trainY)
		sess.run(train, feed_dict = {x : batchX, y : batchY, keep_prob : DROP_OUT})
		if i % 100 == 0:
			# calculating train accuracy
			acc, lossTmp = sess.run([accuracy, loss], feed_dict = {x : batchX, y : batchY, keep_prob : DROP_OUT})
			print('Iter: '+str(i)+' Minibatch_Loss: '+"{:.6f}".format(lossTmp)+' Train_acc: '+"{:.5f}".format(acc))
	for i in range(5):
		# calculating test accuracy
		testBatchX, testBatchY = data.get_next_batch(BATCH_SIZE, testX, testY)
		testAccuracy = sess.run(accuracy, feed_dict = {x : testBatchX, y : testBatchY, keep_prob : DROP_OUT})
		print('test accuracy : ', testAccuracy)
