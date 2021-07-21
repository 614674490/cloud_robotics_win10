import os

import tensorflow as tf

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

CLOUD_ROOT_DIR = os.environ['CLOUD_ROOT_DIR']
print(CLOUD_ROOT_DIR)

