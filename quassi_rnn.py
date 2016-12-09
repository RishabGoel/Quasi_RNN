import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mnist
import numpy as np
from sklearn.datasets import load_digits



def reshape(x):
	s = tf.reshape(x, [-1, 28, 28, 1])
	d = tf.zeros([x.get_shape().as_list()[0], 28, 27, 1], dtype=tf.float32)
	return tf.concat(2,[d, s])


def create_weights(T, n, m):
	weights = {
		"W_z" : tf.Variable(tf.random_normal([n, T, 1, m])),
		"W_f" : tf.Variable(tf.random_normal([n, T, 1, m])),
		"W_o" : tf.Variable(tf.random_normal([n, T, 1, m])),
		"W" : tf.Variable(tf.random_normal([T*m, 10])),
		"h" : tf.Variable(tf.zeros([batch_size, T, m, 1], dtype = tf.float32)),
		"c" : tf.Variable(tf.zeros([batch_size, T, m, 1], dtype = tf.float32))
	}
	return weights


def update(X, weights, b, idx):
	# Convolution operation
	conv_z = tf.nn.conv2d(X, weights[idx]["W_z"], strides = [1, 1, 1, 1], padding = "VALID")

	conv_z_reshaped = tf.nn.tanh(tf.transpose(conv_z, perm = [0,2,3,1]))
	conv_f = tf.nn.conv2d(X, weights[idx]["W_f"], strides = [1, 1, 1, 1], padding = "VALID")
	conv_f_reshaped = tf.nn.sigmoid(tf.transpose(conv_f, perm = [0,2,3,1]))
	conv_o = tf.nn.conv2d(X, weights[idx]["W_o"], strides = [1, 1, 1, 1], padding = "VALID")
	conv_o_reshaped = tf.nn.sigmoid(tf.transpose(conv_o, perm = [0,2,3,1]))
	c_conv = (1-conv_f_reshaped)*conv_z_reshaped

	# Pooling operation
	with tf.device('/cpu:0'):
		for j in range(batch_size):
			if j == 0:
				c = weights[idx]["c"][-1,:,:,:]
			else:
				c = weights[idx]["c"][j-1,:,:,:]
			for i in range(c.get_shape()[1]):
				tmp = c[i,:,:]*conv_f_reshaped[j,i,:,:] + c_conv[j, i, :, :]
				tmp = tf.reshape(tmp,[1,tmp.get_shape().as_list()[0],tmp.get_shape().as_list()[1]])
				if i == 0:
					c = tf.concat(0,[tmp, c[i+1:,:,:]])
				else:
					c = tf.concat(0,[c[:i,:,:],tmp, c[i+1:,:,:]])
			c = tf.reshape(c, [1, c.get_shape().as_list()[0], c.get_shape().as_list()[1], c.get_shape().as_list()[2]])
			if j==0:
				weights[idx]["c"] = tf.concat(0,[c, weights[idx]["c"][j+1:,:,:,:]])
			else:
				weights[idx]["c"] = tf.concat(0,[weights[idx]["c"][:j,:,:,:], c, weights[idx]["c"][j+1:,:,:,:]])
	weights[idx]["h"] = conv_o_reshaped*weights[idx]["c"]

	return weights[idx]["c"]


def quasinet(X, weights, biases):
    layer_1 = update(X, weights, biases, 0)
    weights[0]["c"] = layer_1
    dims = weights[1]["W_z"].get_shape().as_list()
    d = tf.zeros([layer_1.get_shape().as_list()[0], dims[0], dims[1]-1, 1], dtype=tf.float32)
    layer_2 = tf.concat(2,[d, tf.transpose(layer_1, perm=[0,2,1,3])])
    layer_2 = update(layer_2, weights, biases, 1)
    
    
    weights[1]["c"] = layer_2
    dim_prod = layer_2.get_shape().as_list()
    fc1 = tf.reshape(layer_2, [dim_prod[0], dim_prod[1]*dim_prod[2]])
    
    fc1 = tf.nn.tanh(tf.add(tf.matmul(fc1, weights[-1]["W"]), biases[-1]))
    return fc1

learning_rate = 0.0001
training_iters = 200000
batch_size = 128
display_step = 10

mnist_data = mnist.input_data.read_data_sets("/tmp/data", one_hot=True)

X = tf.placeholder(tf.float32,[batch_size, 784])
X_reshaped = reshape(X)
Y = tf.placeholder(tf.float32,[batch_size, 10])

# each element is tuple T, n, m specifying the layer
# layer_params =  [(28, 28, 20), (28, 20, 15)]
layer_params = [(28, 28, 20), (28, 20, 15)]
weights = []
biases = []
for i in range(len(layer_params)):
	weights.append(create_weights(layer_params[i][0],layer_params[i][1], layer_params[i][2]))
	biases.append(tf.Variable(tf.random_normal([10])))

pred = quasinet(X_reshaped, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	# print "B"
	# print weights[0]["h"].eval()

	step = 1
	# Keep training until reach max iterations
	while step * batch_size < training_iters:
	# if step==1:
	# for i in range(100):
		# print weights[0]["h"].eval()
		batch_x, batch_y = mnist_data.train.next_batch(batch_size)
		# Run optimization op (backprop)
		# print weights[0]["h"].eval(),"begin"
		sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
		# print weights[0]["h"].eval(),"updated"
		if step % display_step == 0:
			# Calculate batch loss and accuracy
			loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_x,
	                                                          Y: batch_y})
			print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
	              "{:.6f}".format(loss) + ", Training Accuracy= " + \
	              "{:.5f}".format(acc)
		step += 1
	print "Optimization Finished!"

	# Calculate accuracy for 256 mnist test images
	print "Testing Accuracy:", \
		sess.run(accuracy, feed_dict={X: mnist_data.test.images[:batch_size],
	                                  Y: mnist_data.test.labels[:batch_size]})