import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers
import pandas as pd
import numpy as np
import os
import datetime
import argparse

tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = None

IMG_DIM = 256

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
  
  # For each of the image labels, create a column in the dataframe with a
  # 0 or 1 value indicating whether that label was absent or present in the
  # image's labels
  for word in VOCABULARY:
    labels_df[word] = labels_df['tags'].str.contains(word) * 1
  
  # Construct a full filename, assuming that the script is being run from
  # the directory in which the images are stored
  labels_df['filename'] = labels_df['image_name'].map(
      lambda image_name: os.path.join(FLAGS.images_dir,
                                      image_name + "." + FLAGS.image_extension))
  
  # Split off 20% of the images to use for final evaluation of the model
  evaluation_mask = np.random.rand(len(labels_df)) < 0.2
  evaluation_df = labels_df[evaluation_mask]
  
  # Split the remaining dataset 80%-20% for evaluation during training
  test_train_df = labels_df[~evaluation_mask]
  test_mask = np.random.rand(len(test_train_df)) < 0.2
  test_df = test_train_df[test_mask]
  train_df = test_train_df[~test_mask]
  
  return train_df, test_df, evaluation_df

def make_input_fn(input_df, batch_size=10, num_epochs=20):
  def input_fn():
    # Pull off the filenames from the main dataframe.
    filenames = tf.convert_to_tensor(input_df['filename'].as_matrix(), dtype=tf.string)
    
    # For now, we build a model that only classifies whether or not there is
    # agriculture in the image. Pull off the 'agriculture' labels from the
    # dataframe.
    agriculture = tf.convert_to_tensor(input_df['agriculture'].as_matrix(), dtype=tf.uint8)

    # Join the filenames and labels together into an input queue, so that
    # they can be read in a queue outside of the training loop.
    input_queue = tf.train.slice_input_producer([filenames, agriculture],
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


def summarize_convolution(convolution, width, height, channels, filters, name):
  # Concatenate the filters into one image
  # We start with [N, width, height, channels * filters], ordered by channel first, then by filter
  # We want to get to [N*filters, width, height, channels]
  # So, we first split by channels on the last dimension
  # Then we concatenate back together on the first dimension.
  splits = tf.split(convolution, filters, axis=3) # list of [N, width, height, channels]
  merged = tf.concat(splits, 0)
  tf.summary.image(name, merged, max_outputs=200)

def model_fn(features, target):
  # Transform our target into a one-hot encoding of 
  # TRUE and FALSE categories.
  tf.summary.histogram("target_histogram", target)
  target = tf.one_hot(target, 2)

  net = tf.reshape(features["image"], (-1, IMG_DIM, IMG_DIM, 3))
  tf.summary.histogram("input_image_histogram", net)

  net = layers.repeat(net, 2, layers.conv2d, 64, [3, 3], scope='conv1')
  summarize_convolution(net, IMG_DIM, IMG_DIM, 1, 64, "conv1")
  net = layers.max_pool2d(net, [2, 2], scope='pool1')

  net = layers.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
  summarize_convolution(net, IMG_DIM/2, IMG_DIM/2, 1, 128, "conv2")
  net = layers.max_pool2d(net, [2, 2], scope='pool2')

  net = layers.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
  summarize_convolution(net, IMG_DIM/4, IMG_DIM/8, 1, 256, "conv3")
  net = layers.max_pool2d(net, [2, 2], scope='pool3')

  net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
  summarize_convolution(net, IMG_DIM/8, IMG_DIM/8, 1, 512, "conv4")
  net = layers.max_pool2d(net, [2, 2], scope='pool4')

  net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
  summarize_convolution(net, IMG_DIM/16, IMG_DIM/16, 1, 512, "conv5")
  net = layers.max_pool2d(net, [2, 2], scope='pool5')

  net = layers.flatten(net, scope='flatten5')
  net = layers.fully_connected(net, 4096, scope='fc6')
  net = tf.nn.dropout(net, 0.2)
  net = layers.fully_connected(net, 4096, scope='fc7')
  net = tf.nn.dropout(net, 0.2)
  net = layers.fully_connected(net, 1000, scope='fc8')
  
  prediction, loss = tf.contrib.learn.models.logistic_regression_zero_init(net, target)
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
      learning_rate=FLAGS.learning_rate)
  return {'classes': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op


def train():
  train_df, test_df, eval_df = read_data_labels()
  
  classifier = tf.contrib.learn.Estimator(
    model_fn=model_fn,
    model_dir=os.path.join(FLAGS.model_basedir, datetime.datetime.now().strftime("%y-%m-%d-%H.%M")))
  
  validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
      input_fn=make_input_fn(test_df, batch_size=FLAGS.batch_size),
      eval_steps=FLAGS.eval_steps,
      every_n_steps=FLAGS.eval_every_n_steps,
      metrics={
            "accuracy":
                tf.contrib.learn.MetricSpec(
                    metric_fn=tf.contrib.metrics.streaming_accuracy,
                    prediction_key="classes"),
            "precision":
                tf.contrib.learn.MetricSpec(
                    metric_fn=tf.contrib.metrics.streaming_precision,
                    prediction_key="classes"),
            "recall":
                tf.contrib.learn.MetricSpec(
                    metric_fn=tf.contrib.metrics.streaming_recall,
                    prediction_key="classes"),
        },
      early_stopping_rounds=(5000 if FLAGS.early_stopping else None),
      early_stopping_metric="accuracy",
      early_stopping_metric_minimize=True)
  classifier.fit(input_fn=make_input_fn(train_df, batch_size=FLAGS.batch_size),
                 steps=len(train_df),
                 monitors=[validation_monitor])
  
  classifier.evaluate(input_fn=make_input_fn(eval_df, batch_size=FLAGS.batch_size))


def main(argv):
  with tf.Session():
    with tf.device(FLAGS.device):
      train()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
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
