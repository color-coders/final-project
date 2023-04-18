from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.batch_size = 64
        self.num_classes = 2
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        # TODO: Initialize all hyperparameters
        self.e = 1e-5
        
        # TODO: Initialize all trainable parameters
        def make_variables(*dims, initializer=tf.random.truncated_normal): 
            return tf.Variable(initializer(dims, stddev=0.1, dtype=tf.float32), dtype=tf.float32)
        
        self.w1 = make_variables(80, 64)
        self.b1 = make_variables(64)
        self.w2 = make_variables(64, 32)
        self.b2 = make_variables(32)
        self.w3 = make_variables(32, 2)
        self.b3 = make_variables(2)

        self.filter1 = make_variables(5, 5, 3, 16)
        self.filter2 = make_variables(5, 5, 16, 20)
        self.filter3 = make_variables(3, 3, 20, 20)

        self.bias1 = make_variables(16)
        self.bias2 = make_variables(20)
        self.bias3 = make_variables(20)
        
    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)
        
        inputs = tf.cast(inputs, tf.float32)
        conv1 = tf.nn.conv2d(inputs, self.filter1, [2, 2], padding='SAME')
        c1out = tf.nn.bias_add(conv1, self.bias1)
        mean1, variance1 = tf.nn.moments(c1out, axes=[0,1,2])
        batch1 = tf.nn.batch_normalization(conv1, mean1, variance1, None, None, self.e)
        relu1 = tf.nn.relu(batch1)
        maxpool1 = tf.nn.max_pool(relu1, ksize=3, strides=2, padding="SAME")
        conv2 = tf.nn.conv2d(maxpool1, self.filter2, [2, 2], padding="SAME")
        c2out = tf.nn.bias_add(conv2, self.bias2)
        mean2, variance2 = tf.nn.moments(c2out, axes=[0,1,2])
        batch2 = tf.nn.batch_normalization(conv2, mean2, variance2, None, None, self.e)
        relu2 = tf.nn.relu(batch2)
        maxpool2 = tf.cast(tf.nn.max_pool(relu2, ksize=2, strides=2, padding="SAME"), tf.float32)
        if is_testing:
            conv3 = conv2d(maxpool2, self.filter3, [1, 1, 1, 1], "SAME")
        else:
            conv3 = tf.nn.conv2d(maxpool2, self.filter3, [1, 1], padding="SAME")
        c3out = tf.nn.bias_add(conv3, self.bias3)
        mean3, variance3 = tf.nn.moments(c3out, axes=[0,1,2])
        batch3 = tf.nn.batch_normalization(conv3, mean3, variance3, None, None, self.e)
        relu3 = tf.nn.relu(batch3)

        x = tf.cast(tf.reshape(relu3, [len(inputs), -1]), tf.float32)

        dense1 = tf.matmul(x, self.w1) + self.b1
        rd1 = tf.nn.relu(dense1)
        drop1 = tf.cast(tf.nn.dropout(rd1, rate=0.3), tf.float32)
        dense2 = tf.matmul(drop1, self.w2) + self.b2
        rd2 = tf.nn.relu(dense2)
        drop2 = tf.cast(tf.nn.dropout(rd2, rate=0.3), tf.float32)
        dense3 = tf.matmul(drop2, self.w3) + self.b3

        return dense3

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        softmax = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
        softmax = tf.reduce_mean(softmax)
        return softmax

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels - no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    index = list(range(len(train_inputs)))
    np.random.shuffle(index)
    tf.gather(train_inputs, index)
    tf.gather(train_labels, index)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    b = model.batch_size
    for i in range(0, int((len(train_inputs)/b)-1)):
        xBatch = train_inputs[i*b:(i+1)*b]
        yBatch = train_labels[i*b:(i+1)*b]

        xBatch = tf.image.random_flip_left_right(xBatch)

        with tf.GradientTape() as tape:
            logits = model.call(xBatch, is_testing=False)
            loss = model.loss(logits, yBatch)
            model.loss_list.append(loss)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return model.loss_list


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    acc = 0
    batch_num = 0
    b = model.batch_size
    for i in range(0, int((len(test_inputs)/b)-1)):
        xBatch = test_inputs[i*b:(i+1)*b]
        yBatch = test_labels[i*b:(i+1)*b]

        logits = model.call(xBatch, is_testing=False)
        acc += model.accuracy(logits, yBatch)
        batch_num += 1

    acc /= batch_num
    return acc


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''
    model = Model()

    traininputs, trainlabels = get_data('../data/train', 3, 5)
    testinputs, testlabels = get_data('../data/test', 3, 5)
    
    epochs = 5
    for i in range(epochs):      
        train(model, traininputs, trainlabels)
        test(model, testinputs, testlabels)
        print(f'Epoch {i}')
    
    return


if __name__ == '__main__':
    main()