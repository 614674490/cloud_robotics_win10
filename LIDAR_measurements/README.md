<!--
 * @Author: Ken Kaneki
 * @Date: 2021-07-05 13:10:57
 * @LastEditTime: 2021-07-21 18:33:09
 * @Description: README
-->
## plots for camera-ready

1. source vs receiver WHILE transmitting

    - bw_lidar_source_with_receiver.txt
    - bw_lidar_receiver_with_receiver.txt

    - hz_lidar_source_with_receiver.txt
    - hz_lidar_receiver_with_receiver.txt

2. source vs receiver WITH RVIZ
    - bw_lidar_source_with_receiver.txt
    - bw_lidar_receiver_with_rviz.txt

3. source vs receiver WITH download
    - bw_lidar_source_with_receiver.txt
    - bw_lidar_receiver_with_download.txt

4. source vs receiver with 2 LIDAR sources running
    - bw_lidar_source_with_receiver.txt
    - bw_lidar_receiver_lidar1.txt
    - bw_lidar_receiver_lidar2.txt

## IDEAL PLOT:

- Left (bandwidth), Right (Hz)
- source
- 1 receiver
- 1 receiver with rviz
- 1 receiver with simul file download
- 2 receivers

- bandwidth on right:
    - Source
    - 1 Receiver
    - 1 Receiver + Heavy Background Traffic
    - 2 Concurrent Sender/Receiver Pairs

## velodyne datasheet: 8.6 Mbps

    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6387135/

