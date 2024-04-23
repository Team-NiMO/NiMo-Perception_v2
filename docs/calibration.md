# Camera Calibration
Instructions for eye-on-hand calibration specifically for the Intel Realsense D405 on the UFactory xArm.

## Setup

#### Easy Handeye
Follow the installation instructions for [easy_handeye](https://github.com/IFL-CAMP/easy_handeye).

Add [xArm_realsense.launch](/docs/xArm_realsense.launch) to the easy_handeye repository under `easy_handeye/launch`.

In [xArm_realsense.launch](/docs/xArm_realsense.launch) change `robot_ip` to the ip of the xArm you're working with.

#### ArUco
Print a marker from [ArUco Marker Generator](https://chev.me/arucogen/), be sure to select 'Original ArUco' for the dictionary. Measure the edge length of the marker in meters and update the value in [xArm_realsense.launch](/docs/xArm_realsense.launch).

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

#### Temporary: Until manipulation config is updated to include calibration parameters

The camera projects the depth cloud onto the x-axis, not the z-axis, so the frame needs to be rotated from the calibration values.

First, using [this website](https://www.andre-gaschler.com/rotationconverter/), convert the orientation from quaternion to euler angles. Then, adjust the orientation as follows:

```
x -> x + pi/2
y -> y - pi/2
z -> z
```

In the file `xarm_ros/xarm6_moveit_config/launch/realMove_exec.launch`, add or modify the static transform at the end of the file. Note that the euler angles are zyx, not xyz.

```
<node pkg="tf" type="static_transform_publisher" name="camera_link" args="x y z rz ry rx /link_eef /camera_link 100"/>
```