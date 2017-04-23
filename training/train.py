import tensorflow as tf
import tensorflow.contrib.layers as layers
import pandas as pd
import numpy as np
import os

tf.logging.set_verbosity(tf.logging.INFO)

# Read the training data
labels_df = pd.read_csv("train.csv")

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
labels_df['filename'] = labels_df['image_name'].map(lambda image_name: os.path.join(os.getcwd(), image_name + ".jpg"))

# Split off 20% of the images to use for final evaluation of the model
evaluation_mask = np.random.rand(len(labels_df)) < 0.2
evaluation_df = labels_df[evaluation_mask]

# Split the remaining dataset 80%-20% for evaluation during training
test_train_df = labels_df[~evaluation_mask]
test_mask = np.random.rand(len(test_train_df)) < 0.2
test_df = test_train_df[test_mask]
train_df = test_train_df[~test_mask]

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
    image = tf.image.decode_jpeg(file_content, channels=3)
    image.set_shape([256, 256, 3])

    filenames = input_queue[0]
    labels = input_queue[1]

    # Batch up our input for training according to the requested batch size.
    image_batch, filenames_batch, label_batch = tf.train.batch([image, filenames, labels],
                                                                batch_size=batch_size)
    return {"image": image_batch, "filename": filenames_batch}, label_batch
    
  return input_fn

def model_fn(features, target):
  # Transform our target into a one-hot encoding of 
  # TRUE and FALSE categories.
  target = tf.one_hot(target, 2)

  net = tf.reshape(features["image"], (-1, 256, 256, 3))
  net = tf.to_float(net)

  net = layers.repeat(net, 2, layers.conv2d, 64, [3, 3], scope='conv1')
  net = layers.max_pool2d(net, [2, 2], scope='pool1')
  net = layers.repeat(net, 2, layers.conv2d, 128, [3, 3], scope='conv2')
  net = layers.max_pool2d(net, [2, 2], scope='pool2')
  net = layers.repeat(net, 3, layers.conv2d, 256, [3, 3], scope='conv3')
  net = layers.max_pool2d(net, [2, 2], scope='pool3')
  net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv4')
  net = layers.max_pool2d(net, [2, 2], scope='pool4')
  net = layers.repeat(net, 3, layers.conv2d, 512, [3, 3], scope='conv5')
  net = layers.max_pool2d(net, [2, 2], scope='pool5')
  net = layers.flatten(net, scope='flatten5')
  net = layers.fully_connected(net, 4096, scope='fc6')
  net = tf.nn.dropout(net, 0.5)
  net = layers.fully_connected(net, 4096, scope='fc7')
  net = tf.nn.dropout(net, 0.5)
  net = layers.fully_connected(net, 1000, scope='fc8')
  
  prediction, loss = tf.contrib.learn.models.logistic_regression_zero_init(net, target)
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
      learning_rate=0.1)
  return {'classes': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op

classifier = tf.contrib.learn.Estimator(model_fn=model_fn)

# On a GTX 1060 with 6G RAM, a batch size of 10 started to slow down the main computer. There
# didn't seem to be much affect on images / second pushed through the system.
BATCH_SIZE=5

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    input_fn=make_input_fn(test_df, batch_size=BATCH_SIZE),
    eval_steps=200/BATCH_SIZE,
    every_n_steps=1000/BATCH_SIZE,
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
    early_stopping_rounds=5000,
    early_stopping_metric="accuracy",
    early_stopping_metric_minimize=True)
classifier.fit(input_fn=make_input_fn(train_df, batch_size=BATCH_SIZE),
               steps=len(train_df),
               monitors=[validation_monitor])

classifier.evaluate(input_fn=make_input_fn(evaluation_df, batch_size=BATCH_SIZE))
