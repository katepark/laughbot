#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import math
import random
import os
# uncomment this line to suppress Tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import xrange as range

from rnn_utils import *
import pdb
from time import gmtime, strftime

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    context_size = 0
    num_mfcc_features = 13 #12 mfcc and 1 energy
    num_final_features = num_mfcc_features * (2 * context_size + 1)

    batch_size = 16	
    num_classes = 2 #laugh or no laugh
    num_hidden = 1 #was 128, we only need the "last one", so try with just 1

    num_epochs = 50 #was 50, tune later, look at graph to see if it's enough
    # l2_lambda = 0.0000001
    lr = 1e-2

class RNNModel():
    """
    Implements a recursive neural network with a single hidden layer.
    This network will predict whether a given line of audio isfunny or not.
    """

    def set_num_examples(self, num_examples):
        self.num_valid_examples = tf.cast(num_examples, tf.int32)


    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph:

        inputs_placeholder: Input placeholder tensor of shape (None, None, num_final_features), type tf.float32
        targets_placeholder: Sparse(?) placeholder, type tf.int32. You don't need to specify shape dimension.
        seq_lens_placeholder: Sequence length placeholder tensor of shape (None), type tf.int32
        """
        inputs_placeholder = None
        targets_placeholder = None
        seq_lens_placeholder = None

        ### YOUR CODE HERE (~3 lines)
        inputs_placeholder = tf.placeholder(tf.float32, shape=(None, None, Config.num_final_features))
        targets_placeholder = tf.placeholder(tf.int32, shape=(None))
        seq_lens_placeholder = tf.placeholder(tf.int32, shape=(None))
        ### END YOUR CODE

        self.inputs_placeholder = inputs_placeholder
        self.targets_placeholder = targets_placeholder
        self.seq_lens_placeholder = seq_lens_placeholder


    def create_feed_dict(self, inputs_batch, targets_batch, seq_lens_batch):
        """Creates the feed_dict for the humor recognizer.
        Takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Returns: The feed dictionary mapping from placeholders to values.
        """        
        feed_dict = {} 

        ### YOUR CODE HERE (~3-4 lines)
        feed_dict = {
                    self.inputs_placeholder : inputs_batch,
                    self.targets_placeholder : targets_batch,
                    self.seq_lens_placeholder : seq_lens_batch
        }
        ### END YOUR CODE

        return feed_dict

    def add_prediction_op(self):
        """Applies a GRU RNN over the input data, then an affine layer projection. Steps to complete 
        in this function: 

        - Roll over inputs_placeholder with GRUCell, producing a Tensor of shape [batch_s, max_timestep,
          num_hidden]. 
        - Apply a W * f + b transformation over the data, where f is each hidden layer feature. This 
          should produce a Tensor of shape [batch_s, max_timesteps, num_classes]. Set this result to 
          "logits". 

        Remember:
            * Use the xavier initialization for matrices (W, but not b).
            * W should be shape [num_hidden, num_classes]. num_classes for our dataset is 12
            * tf.contrib.rnn.GRUCell, tf.contrib.rnn.MultiRNNCell and tf.nn.dynamic_rnn are of interest
        """

        logits = None 

        ### YOUR CODE HERE (~10-15 lines)
        cell = tf.contrib.rnn.GRUCell(Config.num_hidden)
        outputs, state = tf.nn.dynamic_rnn(cell, self.inputs_placeholder, self.seq_lens_placeholder, dtype=tf.float32)
        outputsShape = tf.shape(outputs)
        W = tf.get_variable(name="W", shape=[Config.num_hidden, Config.num_classes], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name="b", shape=[Config.num_classes], initializer=tf.zeros_initializer())

        # reshape to 2D
        outputs2D = tf.reshape(outputs, [-1,Config.num_hidden])
        logits = tf.matmul(outputs2D, W) + b

        self.logits2D = logits
        #reshape to output shape
        logits = tf.reshape(logits, shape=[outputsShape[0], outputsShape[1], Config.num_classes])

        ### END YOUR CODE
        self.last_hidden_state = state # TOO: pass these last hidden states as a feature for determining humor
        print('last hidden state', state)
        self.logits = logits


    def add_training_op(self):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables. The Op returned by this
        function is what must be passed to the `sess.run()` call to cause the model to train. For more 
        information, see:

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer
        
        Examples: https://github.com/pbhatnagar3/cs224s-tensorflow-tutorial/blob/master/tensorflow%20MNIST.ipynb
        https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/multilayer_perceptron.ipynb
        """
        optimizer = None 

        # logits [800, 2] -> reshaped logits [16, 50*2] to compare against targets [16]
        logits_shape = tf.shape(self.logits)
        reshaped_logits = tf.reshape(self.logits, shape=[logits_shape[0], logits_shape[1]*logits_shape[2]])
        
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reshaped_logits, labels=self.targets_placeholder))
        optimizer = tf.train.AdamOptimizer(Config.lr).minimize(self.cost) 
        
        # HELP: second argument of argmax is 1 in example github, but error out of range so reshaped targets
        targets2d = tf.reshape(self.targets_placeholder, [tf.shape(self.targets_placeholder)[0],1])

        correct_pred = tf.equal(tf.argmax(reshaped_logits, 1), tf.argmax(targets2d, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.optimizer = optimizer

    
    def add_decoder_and_wer_op(self):
        """Setup the decoder and add the word error rate calculations here. 

        Tip: You will find tf.nn.ctc_beam_search_decoder and tf.edit_distance methods useful here. 
        Also, report the mean WER over the batch in variable wer

        """        
        # decoded_sequence = None 
        # wer = None 

        # decoded_sequence = tf.nn.ctc_beam_search_decoder(self.logits, self.seq_lens_placeholder, merge_repeated=False)[0][0]
        #wer = tf.edit_distance(tf.cast(decoded_sequence, tf.int32), self.targets_placeholder, normalize=True)
        #wer = tf.reduce_mean(wer)
        # tf.summary.scalar("loss", self.loss)
        #tf.summary.scalar("wer", wer)

        # self.decoded_sequence = decoded_sequence
        #self.wer = wer
    

    def add_summary_op(self):
        tf.summary.scalar("cost", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        self.merged_summary_op = tf.summary.merge_all()

    # This actually builds the computational graph 
    def build(self):
        self.add_placeholders()
        self.add_prediction_op()
        self.add_training_op()       
        # self.add_decoder_and_wer_op()
        self.add_summary_op()
        

    def train_on_batch(self, session, train_inputs_batch, train_targets_batch, train_seq_len_batch, train=True):
        # np.reshape(train_targets_batch, (np.shape(train_targets_batch)[0], 1))

        feed = self.create_feed_dict(train_inputs_batch, train_targets_batch, train_seq_len_batch)

        batch_cost, summary, acc = session.run([self.cost,self.merged_summary_op, self.accuracy], feed)
        
        if math.isnan(batch_cost): # basically all examples in this batch have been skipped 
            return 0
        if train:
            _ = session.run([self.optimizer], feed)

        return batch_cost, summary, acc


    def print_results(self, train_inputs_batch, train_targets_batch, train_seq_len_batch):
        train_feed = self.create_feed_dict(train_inputs_batch, train_targets_batch, train_seq_len_batch)
        print("Testing Accuracy:", session.run(self.accuracy, feed_dict=train_feed))
        # TODO add precision, recall, F1 score

        # deleted because we have no decoded sequence
        # train_first_batch_preds = session.run(self.decoded_sequence, feed_dict=train_feed)        
        # compare_predicted_to_true(train_first_batch_preds, train_targets_batch)


    def __init__(self):
        self.build()

#def get_features_from_rnn(): 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', nargs='?', default='./switchboardaudiosmall.train.pkl', type=str, help="Give path to training data")
    parser.add_argument('--val_path', nargs='?', default='./switchboardaudiosmall.val.pkl', type=str, help="Give path to val data")
    parser.add_argument('--save_every', nargs='?', default=None, type=int, help="Save model every x iterations. Default is not saving at all.")
    parser.add_argument('--print_every', nargs='?', default=10, type=int, help="Print some training and val examples (true and predicted sequences) every x iterations. Default is 10")
    parser.add_argument('--save_to_file', nargs='?', default='saved_models/saved_model_epoch', type=str, help="Provide filename prefix for saving intermediate models")
    parser.add_argument('--load_from_file', nargs='?', default=None, type=str, help="Provide filename to load saved model")
    args = parser.parse_args()

    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())

    train_dataset = load_dataset(args.train_path)
    
    val_dataset = load_dataset(args.val_path)

    train_feature_minibatches, train_labels_minibatches, train_seqlens_minibatches = make_batches(train_dataset, batch_size=Config.batch_size)
    val_feature_minibatches, val_labels_minibatches, val_seqlens_minibatches = make_batches(train_dataset, batch_size=len(val_dataset[0]))

    def pad_all_batches(batch_feature_array):
    	for batch_num in range(len(batch_feature_array)):
    		batch_feature_array[batch_num] = pad_sequences(batch_feature_array[batch_num])[0]
    	return batch_feature_array

    train_feature_minibatches = pad_all_batches(train_feature_minibatches)
    val_feature_minibatches = pad_all_batches(val_feature_minibatches)

    num_examples = np.sum([batch.shape[0] for batch in train_feature_minibatches])
    num_batches_per_epoch = int(math.ceil(num_examples / Config.batch_size))
    
    with tf.Graph().as_default():
        model = RNNModel() 
        model.set_num_examples(num_examples)
        init = tf.global_variables_initializer()

        saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session() as session:
            # Initializate the weights and biases
            session.run(init)
            if args.load_from_file is not None:
            	new_saver = tf.train.import_meta_graph('%s.meta'%args.load_from_file, clear_devices=True)
                new_saver.restore(session, args.load_from_file)
            
            train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)

            global_start = time.time()

            step_ii = 0

            for curr_epoch in range(Config.num_epochs):
                total_train_cost = 0
                total_train_acc = 0
                total_train_los = 0
                start = time.time()

                for batch in random.sample(range(num_batches_per_epoch),num_batches_per_epoch):
                    cur_batch_size = len(train_seqlens_minibatches[batch])

                    batch_cost, summary, acc = model.train_on_batch(session, train_feature_minibatches[batch], train_labels_minibatches[batch], train_seqlens_minibatches[batch], train=True)
                    total_train_cost += batch_cost * cur_batch_size
                    total_train_acc += acc * cur_batch_size

                    train_writer.add_summary(summary, step_ii)
                    step_ii += 1 

                    
                train_cost = total_train_cost / num_examples
                train_acc = total_train_acc / num_examples

                # why only train on first batch of val???
                val_batch_cost, _, val_acc = model.train_on_batch(session, val_feature_minibatches[0], val_labels_minibatches[0], val_seqlens_minibatches[0], train=False)

                log = "Epoch {}/{}, train_cost = {:.3f}, train_accuracy = {:.3f}, val_cost = {:.3f}, val_accuracy = {:.3f}, time = {:.3f}"
                print(log.format(curr_epoch+1, Config.num_epochs, train_cost, train_acc, val_batch_cost, val_acc, time.time() - start))

                if args.print_every is not None and (curr_epoch + 1) % args.print_every == 0: 
                    batch_ii = 0
                    model.print_results(train_feature_minibatches[batch_ii], train_labels_minibatches[batch_ii], train_seqlens_minibatches[batch_ii])

                if args.save_every is not None and args.save_to_file is not None and (curr_epoch + 1) % args.save_every == 0:
                	saver.save(session, args.save_to_file, global_step=curr_epoch + 1)
