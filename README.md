# TF_Planet
Tensorflow Experiments with Planet's Amazon Dataset

This software depends only on Tensorflow, which you can get at https://www.tensorflow.org/.

It also depends on retrieving the Planet Kaggle competition dataset, which you can learn more about here: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space

Known issues:
 * The kaggle JPEG dataset is not directly parseable by the Tensorflow decode_jpeg operation (tested on Windows). To work around this, I re-JPEG-encoded every image using Photoshop. If I were working on Linux at the time, I would write a simple shell script, but it's what I had available at the time.
