<!--
 * @Author: Ken Kaneki
 * @Date: 2021-07-05 13:10:57
 * @LastEditTime: 2021-07-21 18:33:25
 * @Description: README
-->
# PURPOSE: implementation of the streaming offloader

## GENERAL STEPS:

1. embed all the known faces into a 128 dim vector using FaceNet, OpenFace

2. create an SVM for facenet edge model

    - SVM takes embedding to a name of a person and confidence

3. load the neural network offloader

4. STREAMING SETTING

	- parse thru either a live or stored video

	- decide whether to offload or not using NN

	- if offload, query the cloud DNN

	- save interesting frames to a file

	- write offloading predictions and decisions to a file [to report accuracy]

	- compute the accuracy with the offloading policy compared to naive

	- create a video of scenarios when we actually offloaded

## CODE:

TO CREATE SVM FOR FACENET ROBOT AND CLOUD MODELS:

1. embed_and_train_SVM.sh

    - calls extract_embeddings.py
    - runs facenet model $FACENET_DNN and writes embeddings to a pickle file

    - train_model.py trains an SVM based on the pkl'd embeddings and saves:
        - SVM model: recognizer.pickle
        - label encoder mapping a name to a label encoding: le.pickle

## TO RUN OFFLOADER

- offloader_socket_networking/

- uses python sockets to communicate between machines and send frames/FaceNet embeddings over a WiFi network

- this takes less overhead than ROS

