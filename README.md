# NiMo-Perception_v2
A method to detect cornstalks, determine their grasp points, and determine their widths for optimal insertion of a small sensor. This repository is an extension of [CMU_Find_Stalk](https://github.com/aaronzberger/CMU_Find_Stalk) by Aaron Berger.

## Installation
First, clone the repository into the `src` folder of your ROS workspace and pull the weights file for Mask R-CNN.
```
git clone git@github.com:Team-NiMO/NiMo-Perception_v2.git
cd NiMo-Perception_v2
git lfs pull
```

Next, install the python requirements
```
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
Then, update the [configuration file](config/default.yaml) if necessary.

If using an Intel RealSense camera, use [this link](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md#installing-the-packages) to install the RealSense SDK. To confirm installation, run `realsense-viewer` in the terminal with the camera connected.

Then clone the [realsense-ros](https://github.com/IntelRealSense/realsense-ros) repository into the `src` folder of your ROS workspace, switch to the `ros1-legacy` branch and make the workspace.

```
git clone git@github.com:IntelRealSense/realsense-ros.git
cd realsense-ros
git checkout ros1-legacy

cd ../../
catkin_make
```

## Use
To run the node, you must launch the camera node and run `StalkDetect.py`. This launch file has been created for for the realsense camera.

```
roslaunch nimo_perception StalkDetect.launch
```

There are two service calls: 

`GetStalks` - Determine a list of suitable stalks in the region
- Inputs:
    - num_frames: The number of frames to run inference on
    - timeout: The timeout (s) to return error
- Returns:
    - success: The success of the operation (SUCCESS / ERROR / REPOSITION)
    - num_frames: The number of frames inference was run on
    - grasp_points: A list of grasp points (position in m and weight) ordered from highest to lowest weight (relative score of the stalk based on confidence and width)

`GetWidth` - Determine the width of the closest stalk
- Inputs:
    - num_frames: The number of frames to run inference on
    - timeout: The timeout (s) to return error
- Returns:
    - success: The success of the operation (SUCCESS / ERROR / REPOSITION)
    - num_frames: The number of frames inference was run on
    - width: The width of the closest stalk (m)

## Visualization

In the [configuration file](/config/default.yaml), there is an argument called `visualize`. This argument determines whether to publish visualization markers to RViz. There are two types of visualization markers published

`stalk_line` - A green line describing the detected stalk

`grasp_point` - A pink sphere describing the selected grasp point on the stalk

<img src="https://github.com/Team-NiMO/NiMo-Perception_v2/blob/main/docs/rviz.png" width="650">

#### Note: Since the arm is mounted upside-down, so is the world frame and all visualizations

There is also an argument in the [configuration file](/config/default.yaml) called `save_images`. This saves the image captured with the masks from the model with the feature points overlayed on top to the folder [outupt](/output).

If only the mask appears without the feature points, this means that the object was detected by the model, but filtered out as a false detection.

<img src="https://github.com/Team-NiMO/NiMo-Perception_v2/blob/main/docs/visualization.png" width="650">

If this folder is not already created, you may have to create it for images to save.

## Other
### Camera calibration
If the camera needs to be calibrated, refer to [calibration.md](docs/calibration.md)

### Future improvements
- Adding more architectures to [models](src/nimo_perception/models)
- Calibration for width detection
- Average stalk objects instead of grasp points

## Common Issues
**`This repository is over its data quota.`**

Alternatively, the model can be downloaded from [this link](https://drive.google.com/file/d/19bDrrN4pFZPGfqd4r-NZjJYxa13hHlI-/view?usp=share_link), replacing the existing model_field_day1.pth in the [weights](/weights/) folder.

**`numpy.ndarray size changed, may indicate binary incompatibility.`**

Upgrade numpy and rerun node

```
pip install --upgrade numpy
```

## Acknowledgements
- [Aaron Berger](https://github.com/aaronzberger) for his work on [CMU_Find_Stalk](https://github.com/aaronzberger/CMU_Find_Stalk) laying the groundwork for this repository
- [Mark (Moonyoung) Lee](https://github.com/markmlee) for his assistance and advice
- Dr. Oliver Kroemer for his assistance and advice
- Dr. George Kantor for his assistance and advice
- The rest of [Team NiMo](https://github.com/Team-NiMO) for their suppport and feedback