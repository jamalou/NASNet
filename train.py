from nasnet import nasnet
import tensorflow as tf
import os

import numpy as np
import image_reader
#from tensorflow.python.platform import gfile
from tensorflow.contrib import slim as slim 
from six.moves import xrange

import image_reader

data_url = "http://download.tensorflow.org/example_images/flower_photos.tgz" 
data_dir = 'flowers_data'


imgen = image_reader.ImageGenerator(data_url, data_dir, validation_percentage=20, batch_size=32)


image_size = (224, 224)
label_count = imgen.classes_count
learning_rate = 1e-3
summaries_dir = '/tmp/retrain_logs'
train_dir = 'train'
start_checkpoint = None
n_epochs = 3
training_steps_per_epoch = imgen.batch_count['training']
validation_steps = imgen.batch_count['validation']

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

tf.logging.set_verbosity(tf.logging.INFO)

# Start a new TensorFlow session.
sess = tf.InteractiveSession(config=config)

image_input = tf.placeholder(
      tf.float32, [None, *image_size, 3], name='fingerprint_input')

with slim.arg_scope(nasnet.nasnet_mobile_arg_scope()): 
    # build nasnet
    logits, end_points = nasnet.build_nasnet_mobile(image_input, label_count, config= nasnet.mobile_imagenet_config())  
# Define loss and optimizer
ground_truth_input = tf.placeholder(
  tf.float32, [None, label_count], name='groundtruth_input')

# Create the back propagation and training evaluation machinery in the graph.
with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_input, logits=logits))
    
tf.summary.scalar('cross_entropy', cross_entropy_mean)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.name_scope('train'), tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(
        learning_rate)
    train_step = slim.learning.create_train_op(cross_entropy_mean, train_op)
#    train_step = tf.train.GradientDescentOptimizer(
#        learning_rate_input).minimize(cross_entropy_mean)
predicted_indices = tf.argmax(logits, 1)
expected_indices = tf.argmax(ground_truth_input, 1)
correct_prediction = tf.equal(predicted_indices, expected_indices)
#confusion_matrix = tf.confusion_matrix(expected_indices, predicted_indices, num_classes=label_count)
evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', evaluation_step)

global_step = tf.train.get_or_create_global_step()
increment_global_step = tf.assign(global_step, global_step + 1)

saver = tf.train.Saver(tf.global_variables())

# Merge all the summaries and write them out to /tmp/retrain_logs (by default)
merged_summaries = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
validation_writer = tf.summary.FileWriter(summaries_dir + '/validation')

tf.global_variables_initializer().run()


# In[16]:


start_step = 1
if start_checkpoint:
    models.load_variables_from_checkpoint(sess, start_checkpoint)
    start_step = global_step.eval(session=sess)
    
tf.logging.info('Training from step: %d ', start_step)

# Save graph.pbtxt.
tf.train.write_graph(sess.graph_def, train_dir,
                   'nasnet.pbtxt')


# Training loop.
best_accuracy = 0
training_steps_max = n_epochs * training_steps_per_epoch
for training_step in xrange(start_step, training_steps_max + 1):
    # Figure out what the current learning rate is.
    # Pull the image samples we'll use for training.
    train_images, train_ground_truth = imgen.next_batch('training', image_size=(224, 224))
    # Run the graph with this batch of training data.
    train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
        [
            merged_summaries, evaluation_step, cross_entropy_mean, train_step,
            increment_global_step
        ],
        feed_dict={
            image_input: train_images,
            ground_truth_input: train_ground_truth
        })
    train_writer.add_summary(train_summary, training_step)
    tf.logging.info('Step #%d: accuracy %.2f%%, cross entropy %f' %
                    (training_step, train_accuracy * 100,
                     cross_entropy_value))
    is_last_step = (training_step == training_steps_max)
    if (training_step % training_steps_per_epoch) == 0 or is_last_step:
        validation_set_size = imgen.set_size('validation')
        total_accuracy = 0.0
        for validation_step in range(validation_steps):
            validation_images, validation_ground_truth = imgen.next_batch('validation', image_size=(224, 224))
            # Run a validation step and capture training summaries for TensorBoard
            # with the `merged` op.
            validation_summary, validation_accuracy = sess.run(
                [merged_summaries, evaluation_step],
                feed_dict={
                    image_input: validation_images,
                    ground_truth_input: validation_ground_truth
                })
            validation_writer.add_summary(validation_summary, training_step)
            batch_size = len(validation_images)
            total_accuracy += (validation_accuracy * batch_size) / validation_set_size
        
        tf.logging.info('Step %d: Validation accuracy = %.2f%% (N=%d)' %
                      (training_step, total_accuracy * 100, validation_set_size))

        # Save the model checkpoint when validation accuracy improves
        if total_accuracy > best_accuracy:
            best_accuracy = total_accuracy
            checkpoint_path = os.path.join(train_dir, 'best', 'NASNet_'+ str(int(best_accuracy*10000)) + '.ckpt')
            tf.logging.info('Saving best model to "%s-%d"', checkpoint_path, training_step)
            saver.save(sess, checkpoint_path, global_step=training_step)
        tf.logging.info('So far the best validation accuracy is %.2f%%' % (best_accuracy*100))
