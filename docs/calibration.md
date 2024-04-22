# Camera Calibration
Instructions for eye-on-hand calibration specifically for the Intel Realsense D405 on the UFactory xArm.

## Setup

#### Easy Handeye
Follow the installation instructions for [easy_handeye](https://github.com/IFL-CAMP/easy_handeye).

Add [xArm_realsense.launch](/docs/xArm_realsense.launch) to the easy_handeye repository under `easy_handeye/launch`.

In [xArm_realsense.launch](/docs/xArm_realsense.launch) change `robot_ip` to the ip of the xArm you're working with.

#### ArUco
Print a marker from [ArUco Marker Generator](https://chev.me/arucogen/), be sure to select 'Original ArUco' for the dictionary. Measure the edge length of the marker in meters and note it down.

Install aruco ros

```
sudo apt-get install ros-noetic-aruco-ros
```

## Use
Make and source the workspace, then run the launch file

```
catkin_make
source devel/setup.bash
roslaunch easy_handeye xArm_realsense.launch
```

Move the arm in manual mode to an angle where the camera can see the aruco marker and hit capture. Repeat this 10-20 times for different arm configurations then hit compute.

TO-DO: Adjusting rotations after calibration