<!--
 * @Author: Ken Kaneki
 * @Date: 2021-07-05 13:10:57
 * @LastEditTime: 2021-07-21 18:34:27
 * @Description: README
-->

```txt
here we simulate an RL agent for offloading

code uses A2C algorithm in tensorflow and python3
```

## CODE DIRECTORY

1. FaceNet_four_action_simulator/

    - Simulation env for facenet video experiments
    - named Four action since action space has 4 actions:
        - use past robot
        - curr robot
        - past cloud
        - curr cloud

    - has the sim environment, benchmark controller utils, and script to compare them all on test traces
