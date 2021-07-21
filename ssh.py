'''
Author: Ken Kaneki
Date: 2021-07-21 13:56:31
LastEditTime: 2021-07-21 13:56:31
Description: README
'''
import os
import time
count = 0
while True:

    os.system('ssh -T git@github.com')
    count = count+1
    print('ssh keeplive', count)
    time.sleep(10)
