import pandas as pd
import numpy as np
import os

labels_df = pd.read_csv("train.csv")

VOCABULARY = set()
for words in labels_df.tags.str.split(' '):
  for word in words:
    VOCABULARY.add(word)

for word in VOCABULARY:
  labels_df[word] = labels_df['tags'].str.contains(word) * 1

labels_df['filename'] = labels_df['image_name'].map(lambda image_name: os.path.join(os.getcwd(), image_name + ".jpg"))

evaluation_mask = np.random.rand(len(labels_df)) < 0.2
evaluation_df = labels_df[evaluation_mask]

test_train_df = labels_df[~evaluation_mask]
test_mask = np.random.rand(len(test_train_df)) < (500/len(test_train_df))
test_df = test_train_df[test_mask]
train_df = test_train_df[~test_mask]

import tensorflow as tf
import tensorflow.contrib.layers as layers

sess = tf.InteractiveSession()
tf.reset_default_graph()

tf.logging.set_verbosity(tf.logging.INFO)

def make_input_fn(input_df, batch_size=10, num_epochs=20):
  def input_fn():
    filenames = tf.convert_to_tensor(input_df['filename'].as_matrix(), dtype=tf.string)
    # Just classifying whether or not an image contains agriculture, for now
    agriculture = tf.convert_to_tensor(input_df['agriculture'].as_matrix(), dtype=tf.uint8)

    input_queue = tf.train.slice_input_producer([filenames, agriculture],
                                                num_epochs, shuffle=False)

    file_content = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(file_content, channels=3)
    image.set_shape([256, 256, 3])

    filenames = input_queue[0]
    labels = input_queue[1]

    image_batch, filenames_batch, label_batch = tf.train.batch([image, filenames, labels],
                                                                batch_size=batch_size)
    return {"image": image_batch, "filename": filenames_batch}, label_batch
    
  return input_fn

def model_fn(features, target):
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
  return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op

classifier = tf.contrib.learn.Estimator(model_fn=model_fn)

BATCH_SIZE=5

validation_metrics = {
    "accuracy":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_accuracy,
            prediction_key="class"),
    "precision":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_precision,
            prediction_key="class"),
    "recall":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_recall,
            prediction_key="class"),
}
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    input_fn=make_input_fn(test_df, batch_size=BATCH_SIZE),
    every_n_steps=5000 / BATCH_SIZE,
    metrics=validation_metrics)
classifier.fit(input_fn=make_input_fn(train_df, batch_size=BATCH_SIZE),
               steps=len(train_df),
               monitors=[validation_monitor])

classifier.evaluate(input_fn=make_input_fn(evaluation_df, batch_size=BATCH_SIZE))
