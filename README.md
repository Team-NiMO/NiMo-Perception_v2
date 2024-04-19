# NiMo-Perception_v2
A method to detect cornstalks, determine their grasp points, and determine their widths for optimal insertion of a small sensor. This repository is an extension of [CMU_Find_Stalk](https://github.com/aaronzberger/CMU_Find_Stalk) by Aaron Berger.

## Installation
First, clone the repository and pull the weights file for Mask R-CNN.
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
Finally, update the [configuration file](config/default.yaml) if necessary.

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
    - success: The success of the 
    - num_frames: The number of frames inference was run on
    - grasp_points: A list of grasp points (position and weight) ordered from highest to lowest weight

`GetWidth` - Determine the width of the closest stalk
- Inputs:
    - num_frames: The number of frames to run inference on
    - timeout: The timeout (s) to return error
- Returns:
    - success: The success of the 
    - num_frames: The number of frames inference was run on
    - width: The width of the closest stalk

## Other
### Future improvements
- Adding more architectures to [models](src/nimo_perception/models)
- Calibration for width detection

## Acknowledgements
- [Aaron Berger](https://github.com/aaronzberger) for his work on [CMU_Find_Stalk](https://github.com/aaronzberger/CMU_Find_Stalk) laying the groundwork for this repository
- [Mark (Moonyoung) Lee](https://github.com/markmlee) for his assistance and advice
- Dr. Oliver Kroemer for his assistance and advice
- Dr. George Kantor for his assistance and advice
- The rest of [Team NiMo](https://github.com/Team-NiMO) for their suppport and feedback