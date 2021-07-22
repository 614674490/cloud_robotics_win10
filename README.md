<!--
 * @Author: Ken Kaneki
 * @Date: 2021-07-05 13:10:57
 * @LastEditTime: 2021-07-22 10:46:33
 * @Description: README
-->

# cloud_robotics

- <font color=#FF000>**请先阅读[环境配置及运行说明文档](./tfenvs/README_env.md)**</font>
- **本程序目前可以在win10和linux下两种系统中运行，但其余的环境配置方法不同，上述说明文档仅给出了win10下相关环境的配置方法**

```txt
A deep reinforcement learning approach to cloud robotics.

NOTE: Before using this repo in python3, make sure to export the base code directory
on your machine to as CLOUD_ROBOTICS_DIR in your bashrc, like:
    export CLOUD_ROOT_DIR='/Users/csandeep/Documents/cloud_robotics'
```

## Key Directories:

- utils: generic utils to plot and load jsons etc.

- simulate_RL: train an RL agent in tensorflow for cloud offloading

- hardware_experiments: run an RL agent on a Jetson TX2 embedded GPU with IP camera

- data: where to put pre-recorded videos or faces of people to train FaceNet DNN models

- DNN_models:
    - pretrained, base vision DNNs to detect faces and the OpenFace models

- scratch_results:
    - where outputs of runs go, such as logfiles and models

## Citations:

```txt
We modified publicly available code and used vision DNNs from:
```

1. pyimagesearch: https://www.pyimagesearch.com/

2. OpenFace project: https://cmusatyalab.github.io/openface/

3. jsonsockets library


## Common problems on the Jetson TX2 when running Tensorflow with CUDA:

```txt
if you get:


2019-03-27 17:20:39.660979: W ./tensorflow/core/common_runtime/gpu/pool_allocator.h:195] could not allocate pinned host memory of size: 2304

add the following, as per:


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

session = tf.Session(config=config, ...)

https://devtalk.nvidia.com/default/topic/1029742/jetson-tx2/tensorflow-1-6-not-working-with-jetpack-3-2/
```
