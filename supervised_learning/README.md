<!--
 * @Author: Ken Kaneki
 * @Date: 2021-07-05 13:10:57
 * @LastEditTime: 2021-07-21 18:34:56
 * @Description: README
-->
## This subdir is experimental. We wanted to:

1. annotate data from the live videos we took
    1. take a video stream and run edge/cloud DNNs ON EVERY FRAME
        - log all relevant information in raw_df/ folder
    2. now we have raw info, lets add the ground-truth, which requires human annotation
        - see data_annotations/ subfolder
    3. now we add an offloading decision rule in a supervised manner
        - this rule needs to be learned by an offloading DNN

2. train a SUPERVISED learning neural network offloader
   - input features:

     - read from offloader_DNN_features/offloader_DNN_input_features.txt

    - columns:
        SVM_confidence
        embedding_distance
        face_confidence
        frame_diff_val
        numeric_prediction
        unknown_flag
        num_detect

   - call run_train_offloader_DNN.sh
   - this calls:
        - python3 train_supervised_learning_offloader.py

   - train/test median percentage errors should be around 10%

    - this saves the neural net in:
    /keras_offloader_DNN/offloader_DNN_model/
        - feature_scaler.save: feature scaler
        - model.offload.yaml: model def
        - weights.offload.h5: model weights

3. quantify the performance on test videos
    1. run test_streaming_offloader_script.sh

    2. calls test_streaming_offload_NN.py

4. deploy on the Jetson TX2 live video camera

5. quantify the results and compare with existing benchmarks
