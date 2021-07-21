<!--
 * @Author: Ken Kaneki
 * @Date: 2021-07-05 13:10:57
 * @LastEditTime: 2021-07-21 18:40:15
 * @Description: README
-->

# FACENET MODELS FROM OPENFACE:

## openface_nn4.small2.v1.t7

- OpenFace facenet "small model" which generates 128-dim embeddings per face

## nn4.v2.t7:

- FaceNet "large" model

## face_detection_model/

- res10_300x300_ssd_iter_140000.caffemodel
    - used to detect presence of faces in video
    - SSD: single shot detector model


## SVM FACENET MODELS WE CREATE:

```txt
see: hardware_experiments/train_facenet_robot_cloud_models/embed_and_train.sh

    {SMALL,LARGE}_CLOUD_SVM:
        - SVM pickled scikit-learn model: recognizer.pickle
        - map from names to numeric labels, such as "bob = 0': le.pickle
        [label encoder]

    SMALL_EDGE_SVM:
        - robot model, analagous to above
```
