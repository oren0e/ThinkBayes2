# Tensorflow basics #
import tensorflow as tf

# Constants
hello = tf.constant('Hello World')
type(hello)

x = tf.constant(100)
type(x)

# Sessions
# create a session which is a class for running a tensorflow operation
sess = tf.Session()  # this is the environment in which operations are being executed
sess.run(hello)
sess.run(x)
type(sess.run(x))

# Operations
x = tf.constant(2)
y = tf.constant(3)
with tf.Session() as sess:
    print('Operations with Constants')
    print('Addition: ', sess.run(x+y))
    print('Subtraction: ', sess.run(x - y))
    print('Multiplication: ', sess.run(x * y))
    print('Division: ', sess.run(x / y))

# Another common object is a placeholder
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
# define operations using tensorflow
add = tf.add(x,y)
sub = tf.subtract(x,y)
mul = tf.multiply(x,y)
d = {x: 20, y: 30}
with tf.Session() as sess:
    print('Operations with Placeholders')
    print('Addition: ', sess.run(add, feed_dict={x:20, y:30}))
    print('Subtraction: ', sess.run(sub, feed_dict=d))
    print('Multiplication ', sess.run(mul, feed_dict=d))

import numpy as np
a = np.array([[5.0,5.0]])  # 1x2
b = np.array([[2.0], [2.0]])  # 2x1

# convert matrices to tensor objects
mat1 = tf.constant(a)
mat2 = tf.constant(b)

# create a matrix multiplication operation
matrix_multi = tf.matmul(mat1, mat2)
with tf.Session() as sess:
    result = sess.run(matrix_multi)  # we don't need feed_dictionary here because we deal with constants
    print(result)

## MNIST part 1 ## (the more manual way of working with TF)
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/Users/orenepshtain/personal/python_and_bayse/ThinkBayes2/python/Refactored_Py_DS_ML_Bootcamp-master/"
                                  "22-Deep Learning/MNIST_data",one_hot=True)
type(mnist)
mnist.train.images.shape  # 55000 images and 784 pixels per image
mnist.train.num_examples
mnist.test.num_examples

import matplotlib.pyplot as plt
mnist.train.images[1].shape
# restructure to 28x28 pixels to reconstruct the original image
plt.imshow(mnist.train.images[1].reshape(28,28), cmap='gist_gray')
plt.show()
plt.imshow(mnist.train.images[1].reshape(784,1), cmap='gist_gray', aspect=0.02)
plt.show()

# Part 2 #
# create a model to classify the digits based on the values of their arrays
x = tf.placeholder(tf.float32,shape=[None,784])  # we are going to pass these images by batches (550000 images is too many to pass at one go)
# So the 'None' in the shape means that we haven't decided yet on the batch size (thus it is a placeholder),
# but we know the size of the vectors which is 784
# the x placeholder is the training data we are passing in, and we are going to have 2 variables - one for the weights and one for the biases
# define our weights
W = tf.Variable(tf.zeros([784,10]))  # 10 possible numbers (10 possible target labels and 784 inputs (pixels)
# biases
b = tf.Variable(tf.zeros([10]))  # add one bias per class
# Note that because we are going to multiply x by W, the second dimension of x is the same as the first dimension of W
# create the y
y = tf.matmul(x,W) + b

# define a loss and an optimizer
y_true = tf.placeholder(tf.float32, shape=[None,10])  # for correct labels (its going to be a label for the batch (we haven't decided on
# yet), and a 10 possible values
# one_hot = True (when we read the data) means that the labels are one-hot encoded
mnist.train.labels[1] # (0,2,3,4...,9)
# cross-entropy (going to take care of gradient descent optimizer and optimize the error)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cross_entropy)
# cross-entropy is how we defined the error and the optimizer is how we are reducing the error

# create a session and run the session
init = tf.global_variables_initializer()  # initializes all the variables in the session (all what we have defined haven't been run yet)
with tf.Session() as sess:
    sess.run(init)
    for step in range(1000): # how many times we want to feed the batches in
        batch_x, batch_y = mnist.train.next_batch(100)  # mnist has an inbuilt method for training batches, this is not realistic normally
        sess.run(train, feed_dict={x:batch_x, y_true:batch_y})  # passing the batches to the placeholders, that's why we put None before

    matches = tf.equal(tf.argmax(y,1), tf.argmax(y_true,1))  # where are we equal based on predicted and true values
    # calculate accuracy
    acc = tf.reduce_mean(tf.cast(matches, tf.float32))

    print(sess.run(acc, feed_dict={x:mnist.test.images, y_true:mnist.test.labels}))


