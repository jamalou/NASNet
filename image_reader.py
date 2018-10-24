import hashlib
import tensorflow as tf
from tensorflow.python.util import compat
from tensorflow.python.platform import gfile
import random
import tarfile
from six.moves import xrange
from six.moves import urllib
import sys

import os
from PIL import Image
import numpy as np


def which_set(filename, validation_percentage):
    """Determines which data partition the file should belong to.
    We want to keep files in the same training or validation sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.
    Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    Returns:
    String, one of 'training' or 'validation'.
    """
    base_name = os.path.basename(filename)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(compat.as_bytes(base_name)).hexdigest()
    percentage_hash = (int(hash_name_hashed, 16) % 100)
    if percentage_hash < validation_percentage:
        result = 'validation'
    else:
        result = 'training'
    return result


# In[4]:


def PIL2array(img):
    """Convert a PIL image to a numpy ndarray
    Args:
    img: PIL image
    Returns:
    Numpy array holding the image pixels' values floats between 0.0 and 1.0.
    """
    return np.array(img.getdata(), np.float32).reshape(img.size[1], img.size[0], 3) / 255.


def load_img_file(filename, size):
    """Loads an image file and returns a .
    Args:
    filename: Path to the .wav file to load.
    Returns:
    Numpy array holding the image pixels' values floats between 0.0 and 1.0.
    """
    return PIL2array(Image.open(filename).resize(size))

class ImageGenerator(object):
    """Handles loading, partitioning, and preparing training data."""
    def __init__(self, data_url, data_dir, validation_percentage, batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.maybe_download_and_extract_dataset(data_url, data_dir)
        self.classes = [element for element in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, element))]
        self.classes_count = len(self.classes)
        self.prepare_data_index(validation_percentage)
        self.batch_index = {'training': 0, 'validation': 0}
        self.batch_count = {key: int(np.ceil(self.set_size(key)/batch_size)) for key in self.batch_index}
        

    def maybe_download_and_extract_dataset(self, data_url, dest_directory):
        """Download and extract dataset tar file.
        If the data set we're using doesn't already exist, this function
        downloads it from the TensorFlow.org website and unpacks it into a
        directory.
        If the data_url is none, don't download anything and expect the data
        directory to contain the correct files already.
        Args:
          data_url: Web location of the tar file containing the data set.
          dest_directory: File path to extract data to.
        """
        if not data_url:
            return
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = data_url.split('/')[-1]
        self.data_dir = os.path.join(dest_directory, filename.split('.')[0])
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):

            def _progress(count, block_size, total_size):
                sys.stdout.write(
                    '\r>> Downloading %s %.1f%%' %
                    (filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            try:
                filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
            except:
                tf.logging.error('Failed to download URL: %s to folder: %s', data_url,
                                 filepath)
                tf.logging.error('Please make sure you have enough free space and'
                                 ' an internet connection')
                raise
            print()
            statinfo = os.stat(filepath)
            tf.logging.info('Successfully downloaded %s (%d bytes)', filename,
                          statinfo.st_size)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
        
    def prepare_data_index(self, validation_percentage):
        """Prepares a list of the samples organized by set and label.
        The training loop needs a list of all the available data, organized by
        which partition it should belong to, and with ground truth labels attached.
        This function analyzes the folders below the `data_dir`, figures out the
        right labels for each file based on the name of the subdirectory it belongs to,
        and uses a stable hash to assign it to a data set partition.
        Args:
          validation_percentage: How much of the data set to use for validation.
        Returns:
          Dictionary containing a list of file information for each set partition,
          and a lookup map for each class to determine its numeric index.
        """
        self.classes_index = {}
        for index, class_ in enumerate(self.classes):
            self.classes_index[class_] = index
        self.data_index = {'validation': [], 'training': []}
        # Look through all the subfolders to find image files
        search_path = os.path.join(self.data_dir, '*', '*.jpg')
        for img_path in gfile.Glob(search_path):
            _, class_ = os.path.split(os.path.dirname(img_path))
            class_ = class_.lower()
            set_index = which_set(img_path, validation_percentage)
            self.data_index[set_index].append({'label': class_, 'file': img_path})
    
    def set_size(self, mode):
        """Calculates the number of samples in the dataset partition.
        Args:
          mode: Which partition, must be 'training' or 'validation'.
        Returns:
          Number of images in the partition.
        """
        return len(self.data_index[mode])

    
    def next_batch(self, mode, image_size=(224, 224)):
        """Gather samples from the data set.
        When the mode is 'training', a random selection of samples will be returned,
        otherwise the first N images in the validation partition will be used.
        Args:
          mode: training or validation
          img_size: the desired image size 
        Returns:
          List of sample data and list of labels in one-hot form.
        """
        assert mode in ['training', 'validation']
        if mode == 'training' and self.batch_index[mode] == 0:
            random.shuffle(self.data_index[mode])
        # Pick one of the partitions to choose samples from.
        candidates = self.data_index[mode]
        # offset: the index to start gathering images from
        offset = self.batch_index[mode] * self.batch_size
        # imgs_count: how many images to gather, useful when the last batch in the data set is smaller than the batch size
        imgs_count = max(0, min(self.batch_size, len(candidates) - offset))

        # Data and labels will be populated and returned.
        data = np.zeros((imgs_count, *image_size, 3))
        labels = np.zeros((imgs_count, self.classes_count))

        for i in xrange(offset, offset + imgs_count):
            sample = candidates[i]
            
            data[i-offset, :] = load_img_file(sample['file'], image_size) 
            label_index = self.classes_index[sample['label']]
            labels[i-offset, label_index] = 1
        
        # increment the batch index to keep track of the already read data
        self.batch_index[mode] = (self.batch_index[mode]+1) % self.batch_count[mode]
        return data, labels



