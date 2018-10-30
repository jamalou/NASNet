from nasnet import nasnet
import tensorflow as tf
import os

import numpy as np
import image_reader
#from tensorflow.python.platform import gfile
from tensorflow.contrib import slim as slim 



def train_network(n_epochs=10, start_checkpoint=None):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        start_epoch = 1

        if start_checkpoint:
            saver.restore(sess, start_checkpoint)
            start_epoch = epoch_number.eval(session=sess) + 1

        tf.logging.info('Training from epoch: %d ', start_epoch)

        # Save graph.pbtxt.
        tf.train.write_graph(sess.graph_def, train_dir, 'nasnet.pbtxt')


        # Training loop.
        best_accuracy = 0
        #training_steps_max = n_epochs * training_steps_per_epoch
        for epoch in range(start_epoch, n_epochs+1):

            #train the model for one epoch
            for training_step in range(1, training_steps_per_epoch+1):
                # Pull the image samples we'll use for training.
                train_images, train_ground_truth = imgen.next_batch('training', image_size=image_size)
                # Run the graph with this batch of training data.
                train_summary, train_accuracy, cross_entropy_value, _ = sess.run(
                    [
                        merged_summaries, evaluation_step, cross_entropy_mean, train_step
                    ],
                    feed_dict={
                        image_input: train_images,
                        ground_truth_input: train_ground_truth
                    })
                train_writer.add_summary(train_summary, (epoch-1)*training_steps_per_epoch+training_step)
                tf.logging.info('Step #%d: accuracy %.2f%%, cross entropy %f' %
                                ((epoch-1)*training_steps_per_epoch+training_step, train_accuracy * 100,
                                 cross_entropy_value))

            #compute validation accuracy
            validation_set_size = imgen.set_size('validation')
            total_accuracy = 0.0
            for validation_step in range(validation_steps):
                validation_images, validation_ground_truth = imgen.next_batch('validation', image_size=image_size)
                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                validation_summary, validation_accuracy = sess.run(
                    [merged_summaries, evaluation_step],
                    feed_dict={
                        image_input: validation_images,
                        ground_truth_input: validation_ground_truth
                    })
                validation_writer.add_summary(validation_summary, epoch*training_steps_per_epoch)
                batch_size = len(validation_images) #the last batch may have different length than the others
                total_accuracy += (validation_accuracy * batch_size) / validation_set_size

            tf.logging.info('Epoch %d: Validation accuracy = %.2f%% (N=%d)' %
                          (epoch, total_accuracy * 100, validation_set_size))

            # Save the model checkpoint when validation accuracy improves
            if total_accuracy > best_accuracy:
                best_accuracy = total_accuracy
                checkpoint_path = os.path.join(train_dir, 'best', 'NASNet_'+ str(int(best_accuracy*10000)) + '.ckpt')
                tf.logging.info('Saving best model to "%s-%d"', checkpoint_path, epoch)
                saver.save(sess, checkpoint_path) #, global_step=training_step//training_steps_per_epoch)

            #save current model anyway
            checkpoint_path = os.path.join(train_dir,'current_checkpoint', 'NASNet.ckpt')
            tf.logging.info('Saving current model to "%s"', checkpoint_path)
            saver.save(sess, checkpoint_path) #, global_step=training_step//training_steps_per_epoch)
            tf.logging.info('So far the best validation accuracy is %.2f%%' % (best_accuracy*100))
            #increment epoch numberf
            sess.run(increment_epoch_number)

    return checkpoint_path

data_url = "" 
data_dir = 'flowers_data/flower_photos'


imgen = image_reader.ImageGenerator(data_url, data_dir, validation_percentage=20, batch_size=32)

image_size = (224, 224)
label_count = imgen.classes_count
learning_rate = 1e-3
summaries_dir = '/tmp/retrain_logs'
train_dir = 'train'
start_checkpoint = None #'train\\best\\NASNet_3218.ckpt-1'
n_epochs = 20
training_steps_per_epoch = imgen.batch_count['training']
validation_steps = imgen.batch_count['validation']

tf.logging.set_verbosity(tf.logging.INFO)

image_input = tf.placeholder(tf.float32, [None, 224, 224, 3], name='image_input')
with slim.arg_scope(nasnet.nasnet_mobile_arg_scope()): 
    # build nasnet
    logits, end_points = nasnet.build_nasnet_mobile(image_input, label_count)  
# Define loss and optimizer
ground_truth_input = tf.placeholder(tf.float32, [None, label_count], name='groundtruth_input')

# Create the back pro4pagation and training evaluation machinery in the graph.
with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=ground_truth_input, logits=logits))

tf.summary.scalar('cross_entropy', cross_entropy_mean)


with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(learning_rate)
    train_step = slim.learning.create_train_op(cross_entropy_mean, train_op)
    #train_step = tf.train.GradientDescentOptimizer(
        #learning_rate).minimize(cross_entropy_mean)
predicted_indices = tf.argmax(logits, 1)
expected_indices = tf.argmax(ground_truth_input, 1)
correct_prediction = tf.equal(predicted_indices, expected_indices)
evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('accuracy', evaluation_step)

epoch_number = tf.Variable(1, name="epoch_number")
increment_epoch_number = tf.assign(epoch_number, epoch_number + 1)


tvars = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if (not "aux" in v.name and not "Aux" in v.name )]
saver = tf.train.Saver(tvars) 
#saver = tf.train.Saver(tf.global_variables())

# Merge all the summaries and write them out to /tmp/retrain_logs (by default)
merged_summaries = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(summaries_dir + '/train')
validation_writer = tf.summary.FileWriter(summaries_dir + '/validation')

ckpt_path = train_network(n_epochs=5, start_checkpoint=None)

tf.reset_default_graph()

image_size = (288, 288)

image_input = tf.placeholder(tf.float32, [None, 288, 288, 3], name='image_input')
with slim.arg_scope(nasnet.nasnet_mobile_arg_scope()): 
    # build nasnet
    logits, end_points = nasnet.build_nasnet_mobile(image_input, label_count)  
# Define loss and optimizer
ground_truth_input = tf.placeholder(tf.float32, [None, label_count], name='groundtruth_input')

# Create the back pro4pagation and training evaluation machinery in the graph.
with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=ground_truth_input, logits=logits))

tf.summary.scalar('cross_entropy', cross_entropy_mean)


with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(learning_rate)
    train_step = slim.learning.create_train_op(cross_entropy_mean, train_op)
    #train_step = tf.train.GradientDescentOptimizer(
        #learning_rate).minimize(cross_entropy_mean)
predicted_indices = tf.argmax(logits, 1)
expected_indices = tf.argmax(ground_truth_input, 1)
correct_prediction = tf.equal(predicted_indices, expected_indices)
evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('accuracy', evaluation_step)

epoch_number = tf.Variable(1, name="epoch_number")
increment_epoch_number = tf.assign(epoch_number, epoch_number + 1)

tvars = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if (not "aux" in v.name and not "Aux" in v.name )]
saver = tf.train.Saver(tvars) 
# Merge all the summaries and write them out to /tmp/retrain_logs (by default)
merged_summaries = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(summaries_dir + '/train')
validation_writer = tf.summary.FileWriter(summaries_dir + '/validation')

train_network(n_epochs=10, start_checkpoint=ckpt_path)

