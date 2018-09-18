# swc_robot
Robot for destroying blue balloons. 

# How to run python script on startup in Raspbery pi 

open and edit `/etc/profile` using: 

```
sudo nano /etc/profile
```

Scroll to the bottom and add: 

```
sudo python <path/to/the/python/script.py
```

For this project, we add: 

```
sudo python /home/pi/Desktop/swc_robot/test_run_robot.py 
```

Note that the camera will continue to run in the background when the 
computer is booted up this way, and so you won't be able to use the camera
for additional code whilst the computer is on.



