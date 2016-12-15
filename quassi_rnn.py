import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mnist
import numpy as np
from sklearn.datasets import load_digits



def reshape(x):
    s = tf.reshape(x, [-1, 8, 8, 1])
    d = tf.zeros([x.get_shape().as_list()[0], 8, 7, 1], dtype=tf.float32)
    return tf.concat(2,[d, s])

def create_weights(T, n, m):
	weights = {
        "W_z" : tf.Variable(tf.random_normal([n, T, 1, m])),
        "W_f" : tf.Variable(tf.random_normal([n, T, 1, m])),
        "W_o" : tf.Variable(tf.random_normal([n, T, 1, m])),
        "W" : tf.Variable(tf.random_normal([T*m, 10])),
        "h" : tf.Variable(tf.zeros([batch_size, T, m, 1], dtype = tf.float32)),
        "c" : tf.Variable(tf.zeros([1, m, 1], dtype = tf.float32))   
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
    c = weights[idx]["c"][0,:,:]
    cell_state = []
    with tf.device('/cpu:0'):
        for j in range(batch_size):
            ind_cell_state = []
            for i in range(c_conv.get_shape().as_list()[1]):
                c = c*conv_f_reshaped[j,i,:,:] + c_conv[j, i, :, :]
                ind_cell_state.append(c)
            ind_cell_state = tf.convert_to_tensor(ind_cell_state)
            dims_ics = ind_cell_state.get_shape().as_list()
            cell_state.append(ind_cell_state)
        cell_state = tf.convert_to_tensor(cell_state)
        weights[idx]["h"] = conv_o_reshaped*cell_state
    return c,weights[idx]["h"]

# create layers
def quasinet(X, weights, biases):
    cell_state, layer_1 = update(X, weights, biases, 0)
    weights[0]["h"] = layer_1
    weights[0]["c"] = cell_state
    dims = weights[1]["W_z"].get_shape().as_list()
    d = tf.zeros([layer_1.get_shape().as_list()[0], dims[0], dims[1]-1, 1], dtype=tf.float32)
    layer_2 = tf.concat(2,[d, tf.transpose(layer_1, perm=[0,2,1,3])])
    cell_state2,layer_2 = update(layer_2, weights, biases, 1)
    weights[1]["h"] = layer_2
    weights[1]["c"] = cell_state2
    dim_prod = layer_2.get_shape().as_list()
    fc1 = tf.reshape(layer_2, [dim_prod[0], dim_prod[1]*dim_prod[2]])
    fc1 = tf.nn.tanh(tf.add(tf.matmul(fc1, weights[-1]["W"]), biases[-1]))
    return fc1

def next_batch(step, digits_data, digits_labels, batch_size):
    X = digits_data[batch_size*(step):(step+1)*batch_size]
    Y = digits_labels[batch_size*(step):(step+1)*batch_size]
    return X,Y
    
learning_rate = 0.01
training_iters = 20000
batch_size = 50
display_step = 5

# mnist_data = mnist.input_data.read_data_sets("/tmp/data", one_hot=True)
digits = load_digits(n_class=10)
digits_data = digits["data"]
digits_labels = digits["target"]
digits_labels_oh = []
for i in range(digits_data.shape[0]):
    a = [0.0]*10
    a[digits_labels[i]] += 1.0
    digits_labels_oh.append(a)
digits_labels = np.array(digits_labels_oh)
# X = tf.placeholder(tf.float32,[batch_size, 784])
X = tf.placeholder(tf.float32,[batch_size, 64])
X_reshaped = reshape(X)
Y = tf.placeholder(tf.float32,[batch_size, 10])

# each element is tuple T, n, m specifying the layer
# layer_params =  [(28, 28, 20), (28, 20, 15)]
layer_params = [(8, 8, 18), (8, 18, 24)]
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
    step = 1
    # Keep training until reach max iterations
    while step <34:
        batch_x, batch_y = next_batch(step,digits_data, digits_labels, batch_size)
        # Run optimization op (backprop)
        p,q = sess.run([pred,optimizer], feed_dict={X: batch_x, Y: batch_y})
        # print p[0], batch_y[0]
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            preds,loss, acc = sess.run([pred,cost, accuracy], feed_dict={X: batch_x,
                                                              Y: batch_y})
            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
            
        step += 1
    print "Optimization Finished!"

    # Calculate accuracy for batch_size digits
    val = 0.0
    for i in range(30):
        s,d = next_batch(i,digits_data, digits_labels, batch_size)
        
        f=(sess.run(accuracy, feed_dict={X: s,Y: d}))
        # print f
        val+=f*50

    print val/(1500.0)