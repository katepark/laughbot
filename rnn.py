#  Compatibility imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import datetime
import argparse
import math
import random
import os
# uncomment this line to suppress Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from six.moves import xrange as range
import sklearn.metrics as metrics
from languagemodel import  *
from laughbot_realtime import *
from convertaudiosample import *

from rnn_utils import *
import pdb
from time import gmtime, strftime
from adamax import AdamaxOptimizer

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    context_size = 0
    num_mfcc_features = 13 #12 mfcc and 1 energy
    num_final_features = num_mfcc_features * (2 * context_size + 1)
    num_timesteps = 50

    batch_size = 16	
    num_classes = 2 #laugh or no laugh
    num_hidden = 100

    num_epochs = 25 #was 50, tune later, look at graph to see if it's enough
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
        self.last_hidden_state = state # TODO: pass these last hidden states as a feature for determining humor
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

        # logits [16, 50, 2] -> grabbing last timestep to compare logits [16, 2] against targets [16]
        logits_shape = tf.shape(self.logits)
        reshaped_logits = tf.slice(self.logits, [0, logits_shape[1] - 1, 0], [-1, 1, -1])
        reshaped_logits = tf.reshape(reshaped_logits, shape=[logits_shape[0], logits_shape[2]])
        # reshaped_logits = tf.reshape(self.logits, shape=[logits_shape[0], logits_shape[1]*logits_shape[2]])

        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reshaped_logits, labels=self.targets_placeholder))
        optimizer = AdamaxOptimizer(Config.lr).minimize(self.cost) 
        
        # TODO: IS LOGITS[0] LAUGHTER OR LOGITS[1]????

        self.pred = tf.argmax(reshaped_logits, 1)
        correct_pred = tf.equal(tf.argmax(reshaped_logits, 1), tf.cast(self.targets_placeholder, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.optimizer = optimizer

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

        batch_cost, summary, acc, pred, acoustic = session.run([self.cost, self.merged_summary_op, self.accuracy, self.pred, self.last_hidden_state], feed)
        
        if math.isnan(batch_cost): # basically all examples in this batch have been skipped 
            return 0
        if train:
            _ = session.run([self.optimizer], feed)

        return batch_cost, summary, acc, pred, acoustic

    def __init__(self):
        self.build()

def train_language_model(acoustic_features, val_acoustic):
    # print('final train acoustic', acoustic_features[:10])
    # print('final val acoustic', val_acoustic[:10])
    trainExamples = util.readExamples('switchboardsampleL.train')
    valExamples = util.readExamples('switchboardsampleL.val')
    testExamples = util.readExamples('switchboardsampleL.test')
    # trainExamples = util.readExamples('switchboardsamplesmall.train')
    # valExamples = util.readExamples('switchboardsamplesmall.val')
    # testExamples = util.readExamples('switchboardsamplesmall.test')
    # comment for test
    compareExamples = valExamples
    # uncomment for test
    # compareExamples = testExamples
    print('TRAIN MODEL')
    vocabulary, freq_col_idx, regr = learnPredictor(trainExamples, acoustic_features)
    print('TRAIN BASELINE')
    allPosNegBaseline(trainExamples)
    
    print('TEST MODEL')
    testPredictor(compareExamples, val_acoustic)
    print('TEST BASELINE')
    allPosNegBaseline(compareExamples)

    # realtimePredict(vocabulary, freq_col_idx, regr)

def predict_laughter(acoustic):
    predictExamples = util.readExamples('laughbot_text.txt')
    # place holder, call annie's acoustic extractor!
    #sample_acoustic = np.zeros((len(predictExamples), Config.num_hidden))

    prediction = predictLaughter(predictExamples, acoustic)
    return prediction




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', nargs='?', default='./switchboardaudioL.train.pkl', type=str, help="Give path to training data")
    parser.add_argument('--val_path', nargs='?', default='./switchboardaudioL.val.pkl', type=str, help="Give path to val data")
    parser.add_argument('--test_path', nargs='?', default='./switchboardaudioL.test.pkl', type=str, help="Give path to test data")
    parser.add_argument('--save_every', nargs='?', default=Config.num_epochs, type=int, help="Save model every x iterations. Default is not saving at all.")
    parser.add_argument('--print_every', nargs='?', default=10, type=int, help="Print some training and val examples (true and predicted sequences) every x iterations. Default is 10")
    parser.add_argument('--save_to_file', nargs='?', default='saved_models', type=str, help="Provide filename prefix for saving intermediate models")
    parser.add_argument('--load_from_file', nargs='?', default=None, type=str, help="Provide filename to load saved model")
    parser.add_argument('--laugh', nargs='?', default=None, type=str, help="Set to string to call laugh")
    args = parser.parse_args()

    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())


    def pad_all_batches(batch_feature_array):
    	for batch_num in range(len(batch_feature_array)):
    		batch_feature_array[batch_num] = pad_sequences(batch_feature_array[batch_num])[0]
    	return batch_feature_array

    with tf.Graph().as_default():
        model = RNNModel() 
        init = tf.global_variables_initializer()

        saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session() as session:
            # Initializate the weights and biases
            if args.load_from_file is not None:
                print("Reading model parameters from",args.load_from_file)
            	new_saver = tf.train.import_meta_graph('%s.meta'%args.load_from_file, clear_devices=True)
                new_saver.restore(session, args.load_from_file)
                
                if args.laugh is not None:
                    response = raw_input("Press 's' to start: ")
                    while response != 'q':#(x==1): #endless loop mode! replace with x==1 for a one-time test
                        print("press enter to stop recording")
                        record_audio()
                        print("audio recorded")
                        transcript = get_transcript_from_file()
                        print("transcript: ", transcript)
                        convert_audio_sample()
                        
                        test_dataset = load_dataset("laughbot_audio.test.pkl")
                        feature_b, label_b, seqlens_b = make_batches(test_dataset, batch_size=len(test_dataset[0]))
                        feature_b = pad_all_batches(feature_b)
                        batch_cost, summary, acc, predicted, acoustic = model.train_on_batch(session, feature_b[0], label_b[0], seqlens_b[0], train=False)
                        prediction = predict_laughter(acoustic)
                        print('Prediction', prediction)
                        if prediction[0] == 1:
                            playLaughtrack()
                        response = raw_input("Press 'c' to continue, 'q' to quit: ")

                    print('Thanks for talking to me')
                else:
                    print('Running saved model on test set')
                    test_dataset = load_dataset(args.test_path)
                    feature_b, label_b, seqlens_b = make_batches(test_dataset, batch_size=len(test_dataset[0]))
                    feature_b, label_b, seqlens_b = make_batches(test_dataset, batch_size=len(test_dataset[0]))
                    feature_b = pad_all_batches(feature_b)
                    batch_cost, summary, acc, predicted, acoustic = model.train_on_batch(session, feature_b[0], label_b[0], seqlens_b[0], train=False)
                    total_test_acoustic_features = np.array(acoustic)

                    actual = np.array(label_b[0])
                    true_positives = np.count_nonzero(predicted * actual)
                    true_negatives = np.count_nonzero((predicted - 1) * (actual - 1))
                    false_positives = np.count_nonzero(predicted * (actual - 1))
                    false_negatives = np.count_nonzero((predicted - 1) * actual)

                    acc2 = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives > 0) else 0
                    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives > 0) else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall > 0) else 0
                    
                    log = "TEST test_cost = {:.3f}, test_accuracy = {:.3f}"
                    print(log.format(batch_cost, acc2))

                    log_f1 = "TEST   true_pos = {:d}, true_neg = {:d}, false_pos = {:d}, false_neg = {:d}, precision = {:.3f}, recall = {:.3f}, f1 = {:.3f}"
                    print(log_f1.format(true_positives, true_negatives, false_positives, false_negatives, precision, recall, f1))
                    
                    testExamples = util.readExamples('switchboardsampleL.test')
                    testPredictor(testExamples, acoustic)
                    allPosNegBaseline(testExamples)
                
            else:
                print("Created model with fresh parameters")
                
                # IF TRYING TO GET PREVIOUS NUMBERS OF TRAIN AND VAL
                # print("Reading model parameters from",args.load_from_file)
                #new_saver = tf.train.import_meta_graph('saved_models/model.meta', clear_devices=True)
                #new_saver.restore(session, "saved_models/model")
                
                session.run(init)

                train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)

                global_start = time.time()

                step_ii = 0

                train_dataset = load_dataset(args.train_path)
                
                val_dataset = load_dataset(args.val_path)

                train_feature_minibatches, train_labels_minibatches, train_seqlens_minibatches = make_batches(train_dataset, batch_size=Config.batch_size)
                val_feature_minibatches, val_labels_minibatches, val_seqlens_minibatches = make_batches(val_dataset, batch_size=len(val_dataset[0]))

                train_feature_minibatches = pad_all_batches(train_feature_minibatches)
                val_feature_minibatches = pad_all_batches(val_feature_minibatches)

                num_examples = np.sum([batch.shape[0] for batch in train_feature_minibatches])
                num_batches_per_epoch = int(math.ceil(num_examples / Config.batch_size))
                
                val_num_examples = np.sum([batch.shape[0] for batch in val_feature_minibatches])
                val_num_batches_per_epoch = int(math.ceil(val_num_examples / len(val_dataset[0])))
                # model.set_num_examples(num_examples)

                print('TRAIN: ', 'num_ex', num_examples, 'num batches per epoch', num_batches_per_epoch, 'len of seq lens', len(train_seqlens_minibatches), 'len of labels', len(train_labels_minibatches))
                print('VAL: ', 'num_ex', val_num_examples, 'num batches per epoch', val_num_batches_per_epoch, 'len of seq lens', len(val_seqlens_minibatches), 'len of labels', len(val_labels_minibatches))

                for curr_epoch in range(Config.num_epochs):
                    total_train_cost = 0.0
                    total_train_acc = 0.0
                    # total_train_los = 0.0
                    true_positives = 0
                    false_positives = 0
                    false_negatives = 0
                    true_negatives = 0
                    total_acoustic_features = []
                    start = time.time()

                    seq = range(num_batches_per_epoch)
                    
                    if curr_epoch == Config.num_epochs:
                        seq = range(num_batches_per_epoch)
                    else:
                        seq = random.sample(range(num_batches_per_epoch),num_batches_per_epoch)
                    
                    for batch in seq:
                        cur_batch_size = len(train_seqlens_minibatches[batch])
                        batch_cost, summary, acc, predicted, acoustic = model.train_on_batch(session, train_feature_minibatches[batch], train_labels_minibatches[batch], train_seqlens_minibatches[batch], train=False)
                        
                        for example in np.array(acoustic):
                            total_acoustic_features.append(np.array(example))

                        total_train_cost += batch_cost * cur_batch_size
                        total_train_acc += acc * cur_batch_size
                        actual = np.array(train_labels_minibatches[batch])
                        true_positives += np.count_nonzero(predicted * actual)
                        true_negatives += np.count_nonzero((predicted - 1) * (actual - 1))
                        false_positives += np.count_nonzero(predicted * (actual - 1))
                        false_negatives += np.count_nonzero((predicted - 1) * actual)
                        # TODO: change to log correct accuracy after each epoch?
                        # train_writer.add_summary(summary, step_ii)
                        step_ii += 1 

                    train_cost = (total_train_cost) / num_examples
                    train_acc = (total_train_acc) / num_examples

                    train_acc2 = (true_positives + true_negatives) / (true_negatives + true_positives + false_positives + false_negatives)
                    train_precision = (true_positives) / (true_positives + false_positives) if (true_positives + false_positives > 0) else 0
                    train_recall = (true_positives) / (true_positives + false_negatives) if (true_positives + false_negatives > 0) else 0
                    train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall) if (train_precision + train_recall > 0) else 0

                    # val_batch_cost, _, val_acc, val_predicted, val_acoustic = model.train_on_batch(session, val_feature_minibatches[0], val_labels_minibatches[0], val_seqlens_minibatches[0], train=False)
                    total_val_cost, _, total_val_acc, val_predicted, val_acoustic = model.train_on_batch(session, val_feature_minibatches[0], val_labels_minibatches[0], val_seqlens_minibatches[0], train=False)
                    
                    total_val_acoustic_features = np.array(val_acoustic)

                    actual = np.array(val_labels_minibatches[0])
                    val_true_positives = np.count_nonzero(val_predicted * actual)
                    val_true_negatives = np.count_nonzero((val_predicted - 1) * (actual - 1))
                    val_false_positives = np.count_nonzero(val_predicted * (actual - 1))
                    val_false_negatives = np.count_nonzero((val_predicted - 1) * actual)

                    val_acc2 = (val_true_positives + val_true_negatives) / (val_true_positives + val_true_negatives + val_false_positives + val_false_negatives)
                    val_precision = val_true_positives / (val_true_positives + val_false_positives) if (val_true_positives + val_false_positives > 0) else 0
                    val_recall = val_true_positives / (val_true_positives + val_false_negatives) if (val_true_positives + val_false_negatives > 0) else 0
                    val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall) if (val_precision + val_recall > 0) else 0
                    
                    log = "Epoch {}/{}, train_cost = {:.3f}, train_accuracy = {:.3f}, val_cost = {:.3f}, val_accuracy = {:.3f}, time = {:.3f}"
                    print(log.format(curr_epoch+1, Config.num_epochs, train_cost, train_acc2, total_val_cost, val_acc2, time.time() - start))

                    log_f1 = "TRAIN true_pos = {:d}, true_neg = {:d}, false_pos = {:d}, false_neg = {:d}, precision = {:.3f}, recall = {:.3f}, f1 = {:.3f}"
                    print(log_f1.format(true_positives, true_negatives, false_positives, false_negatives, train_precision, train_recall, train_f1))

                    log_f1 = "VAL   true_pos = {:d}, true_neg = {:d}, false_pos = {:d}, false_neg = {:d}, precision = {:.3f}, recall = {:.3f}, f1 = {:.3f}"
                    print(log_f1.format(val_true_positives, val_true_negatives, val_false_positives, val_false_negatives, val_precision, val_recall, val_f1))

                if args.save_to_file is not None:
                    save_path = os.path.join(args.save_to_file, "{:%Y%m%d_%H%M%S}/".format(datetime.datetime.now()))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    saver.save(session, save_path + "model")

                print('---Running language model----')
                train_language_model(total_acoustic_features, total_val_acoustic_features)

