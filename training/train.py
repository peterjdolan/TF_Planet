import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers
import pandas as pd
import numpy as np
import math
import os
import datetime
import argparse

tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = None

IMG_DIM = 256
N_CLASSES = 0

def read_data_labels():
  """Reads the data labels from the 'train.csv' file, and returns processed
  and split dataframes.
  
  Returns: training dataframe, test dataframe, and evaluation dataframe.
  """
  # Read the training data
  labels_df = pd.read_csv(FLAGS.train_csv)
  
  # Construct a vocabulary set of all image labels
  VOCABULARY = set()
  for words in labels_df.tags.str.split(' '):
    for word in words:
      VOCABULARY.add(word)
  VOCABULARY = sorted(list(VOCABULARY))
  print("vocabulary", VOCABULARY)
  
  # HACK: set a global to track the size of the vocabulary
  global N_CLASSES
  N_CLASSES = len(VOCABULARY)
  
  # For each of the image labels, create a column in the dataframe with a
  # 0 or 1 value indicating whether that label was absent or present in the
  # image's labels.
  for word in VOCABULARY:
    labels_df[word] = labels_df['tags'].str.contains(word)
    
  # Add a pair of ones and zeros to a list of binary pairs, where if
  # a label is present for the image, we represent it as 0 - 1 (the "on" class
  # is true), and if it is not then we represent it as 1 - 0 (the "off" class
  # is true)
  def categories(row):
    categories = []
    for word in VOCABULARY:
      if row[word]:
        categories.extend((0,1))
      else:
        categories.extend((1,0))
    return categories
  labels_df['categories'] = labels_df.apply(lambda row: categories(row), axis=1)
  
  # Construct a full filename, assuming that the script is being run from
  # the directory in which the images are stored
  labels_df['filename'] = labels_df['image_name'].map(
      lambda image_name: os.path.join(FLAGS.images_dir,
                                      image_name + "." + FLAGS.image_extension))
    
  return labels_df


def split_df(labels_df):
  # Split off 20% of the images to use for final evaluation of the model
  evaluation_mask = np.random.rand(len(labels_df)) < 0.2
  evaluation_df = labels_df[evaluation_mask]
  
  # Split the remaining dataset 80%-20% for evaluation during training
  test_train_df = labels_df[~evaluation_mask]
  test_mask = np.random.rand(len(test_train_df)) < 0.2
  test_df = test_train_df[test_mask]
  train_df = test_train_df[~test_mask]
  
  return train_df, test_df, evaluation_df


def make_input_fn(input_df, batch_size=10, num_epochs=None):
  def input_fn():
    # Pull off the filenames from the main dataframe.
    filenames = tf.convert_to_tensor(input_df['filename'].as_matrix(), dtype=tf.string)

    # HACK: flatten the lists in the input_df['categories'] manually, then reshape the tensor
    all_category_values_in_one_list = \
        [item for sublist in input_df['categories'] for item in sublist]
    categories = tf.convert_to_tensor(all_category_values_in_one_list, dtype=tf.uint8)
    categories = tf.reshape(categories, [-1, N_CLASSES * 2], "input")

    # Join the filenames and labels together into an input queue, so that
    # they can be read in a queue outside of the training loop.
    input_queue = tf.train.slice_input_producer([filenames, categories],
                                                num_epochs, shuffle=False)

    # Transform the filenames tensor into a decoded JPEG tensor, reading
    # from disk as we go.
    file_content = tf.read_file(input_queue[0])
    image = tf.image.decode_image(file_content, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image.set_shape([IMG_DIM, IMG_DIM, 3])

    filenames = input_queue[0]
    labels = input_queue[1]

    # Batch up our input for training according to the requested batch size.
    image_batch, filenames_batch, label_batch = tf.train.batch([image, filenames, labels],
                                                                batch_size=batch_size)
    return {"image": image_batch, "filename": filenames_batch}, label_batch
    
  return input_fn


# Taken from https://gist.github.com/kukuruza/03731dc494603ceab0c5
def put_kernels_on_grid(kernel, name, pad = 1):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      pad:               number of black pixels around each filter (between them)
    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(math.sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x7
  

def summarize_convolution(net, dimension, name):
  first_image = tf.slice(net, [0, 0, 0, 0], [1, -1, -1, -1])
  squeezed = tf.squeeze(first_image)
  # Here we assume that the convolution kernels are single-channel
  by_channel = tf.reshape(squeezed, [dimension, dimension, 1, -1])
  grid = put_kernels_on_grid(by_channel, name)
  tf.summary.image(name, grid, max_outputs=1)


def model_fn(features, target):
  net = tf.reshape(features["image"], (-1, IMG_DIM, IMG_DIM, 3))
  tf.summary.image('features', net, max_outputs=1)
  tf.summary.histogram("features_histogram", net)

  net = layers.repeat(net, 2, layers.conv2d, 64, [3, 3], scope='conv1')
  summarize_convolution(net, 256, "conv1")
  net = layers.max_pool2d(net, [2, 2], scope='pool1')

  net = layers.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
  summarize_convolution(net, 128, "conv2")
  net = layers.max_pool2d(net, [2, 2], scope='pool2')

  net = layers.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
  summarize_convolution(net, 64, "conv3")
  net = layers.max_pool2d(net, [2, 2], scope='pool3')

  net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
  summarize_convolution(net, 32, "conv4")
  net = layers.max_pool2d(net, [2, 2], scope='pool4')

  net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
  summarize_convolution(net, 16, "conv5")
  net = layers.max_pool2d(net, [2, 2], scope='pool5')

  net = layers.flatten(net, scope='flatten5')
  net = layers.fully_connected(net, 4096, scope='fc6')
  net = tf.nn.dropout(net, 0.2)
  net = layers.fully_connected(net, 4096, scope='fc7')
  net = tf.nn.dropout(net, 0.2)
  net = layers.fully_connected(net, 1000, scope='fc8')
  net = tf.nn.dropout(net, 0.2)
  
  # The final layer, which stores the (not present) - (present) binary pairs
  # for each label in the vocabulary
  prediction = layers.fully_connected(net, N_CLASSES * 2, scope='pred')
  
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,
                                                                labels=target))
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
      learning_rate=FLAGS.learning_rate)
  return {'classes': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op


def train():
  train_df, test_df, eval_df = split_df(read_data_labels())
  
  classifier = tf.contrib.learn.Estimator(
    model_fn=model_fn,
    model_dir=os.path.join(FLAGS.model_basedir, datetime.datetime.now().strftime("%y-%m-%d-%H.%M")),
    config=tf.contrib.learn.RunConfig(
        save_checkpoints_secs=60,
        save_summary_steps=100))
  
  validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
      input_fn=make_input_fn(test_df, batch_size=FLAGS.batch_size),
      eval_steps=FLAGS.eval_steps,
      every_n_steps=FLAGS.eval_every_n_steps)
  classifier.fit(input_fn=make_input_fn(train_df, batch_size=FLAGS.batch_size),
                 steps=len(train_df),
                 monitors=[validation_monitor])
  
  classifier.evaluate(input_fn=make_input_fn(eval_df, batch_size=FLAGS.batch_size))


def main(argv):
  if FLAGS.summarize_input:
    print(read_data_labels())
    return
  
  with tf.Session():
    with tf.device(FLAGS.device):
      train()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--summarize_input',
                      type=bool,
                      default=False,
                      help='Summarize the input dataframe and exit.')
  
  parser.add_argument('--model_basedir', 
                      type=str, 
                      default=os.getcwd(), 
                      help='Base directory in which to store model files')
  parser.add_argument('--train_csv', 
                      type=str, 
                      default='train.csv', 
                      help='Location of train.csv file')
  parser.add_argument('--images_dir', 
                      type=str, 
                      default='./', 
                      help='Location of image files')
  parser.add_argument('--image_extension', 
                      type=str, 
                      default='jpg', 
                      help='Filename extension of image files')
  parser.add_argument('--learning_rate', 
                      type=float, 
                      default=0.1, 
                      help='Learning rate')
  # On a GTX 1060 with 6G RAM, a batch size of 10 started to slow down the main computer. There
  # didn't seem to be much affect on images / second pushed through the system.
  parser.add_argument('--batch_size', 
                      type=int, 
                      default=5, 
                      help='Batch size for training and evaluation')
  parser.add_argument('--train_steps', 
                      type=int,
                      default=10000,
                      help='Number of training steps')
  parser.add_argument('--eval_steps', 
                      type=int,
                      default=100,
                      help='Number of test steps')
  parser.add_argument('--eval_every_n_steps', 
                      type=int,
                      default=500,
                      help='Number of training steps between testing')
  parser.add_argument('--early_stopping',
                      type=bool,
                      default=False,
                      help='Enable or disable early stopping, when accuracy hasn\'t changed for more than 5000 steps.')
  parser.add_argument('--device',
                      type=str,
                      default='/gpu:0',
                      help='The device to run on.')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
