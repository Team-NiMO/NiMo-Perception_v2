import rospy
import tf2_ros
import numpy as np
import tf_conversions
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Point, Pose

def isCameraRunning(camera_info_topic, timeout=2):
        '''
        Determine whether the camera is running by checking the camera_info topic

        Parameters
            timeout: The timeout to wait for the camera info

        Returns
            width: The width of the RGB-D image
            height: The height of the RGB-D image
            intrinsic: The camera intrinsics
        '''
        cameraRunning = True

        try:
            rospy.wait_for_message(camera_info_topic, CameraInfo, timeout=timeout)    
        except rospy.ROSException:
            cameraRunning = False

        return cameraRunning

def getCameraInfo(camera_info_topic, timeout=2):
    '''
    Determine whether the camera is running by checking the camera_info topic

    Parameters
        timeout: The timeout to wait for the camera info

    Returns
        width: The width of the RGB-D image
        height: The height of the RGB-D image
        intrinsic: The camera intrinsics
    '''
    
    try:
        camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo, timeout=timeout)

        width = camera_info.width
        height = camera_info.height
        intrinsic = np.array(camera_info.K).reshape((3, 3))
    except rospy.ROSException:
        return None, None, None, 

    return width, height, intrinsic

def getCam2WorldTransform(camera_frame, world_frame):
    '''
        Get the transform from the camera frame to the world frame

        Parameters
            camera_frame: The name of the camera frame
            world_frame: The name of the world frame

        Returns
            E_cam_to_world: The transformation matrix from the camera frame to the world frame
    '''
    tfBuffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tfBuffer)

    cam_to_world = tfBuffer.lookup_transform(world_frame, camera_frame, rospy.Time(0), rospy.Duration.from_sec(0.5)).transform

    # Convert the Transform msg to a Pose msg
    pose = Pose(position=Point(
                x=cam_to_world.translation.x, y=cam_to_world.translation.y, z=cam_to_world.translation.z),
                orientation=cam_to_world.rotation)

    return tf_conversions.toMatrix(tf_conversions.fromMsg(pose))

def transformCam2World(point, intrinsic, camera_frame, world_frame):
        '''
        Transform a point from the camera frame to the robot frame

        Parameters
            point: The point in the camera frame
            intrinsic: The camera intrinsics
            camera_frame: The name of the camera frame
            world_frame: The name of the world frame

        Returns
            point (x, y, z): The point in the world frame
        '''
        E_cam_to_world = getCam2WorldTransform(camera_frame, world_frame)

        x, y, z = point

        # Normalize the point
        x = (x - intrinsic[0, 2]) / intrinsic[0, 0]
        y = (y - intrinsic[1, 2]) / intrinsic[1, 1]

        # Scale with depth
        x *= z
        y *= z

        # Transform
        x, y, z, _ = np.matmul(E_cam_to_world, np.array([z, x, y, 1]))

        return x, y, z
