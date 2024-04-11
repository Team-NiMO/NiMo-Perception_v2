import yaml
import rospy
import rospkg
import numpy as np
import message_filters

from sensor_msgs.msg import CameraInfo, Image
from nimo_perception.srv import *
from nimo_perception.models import Mask_RCNN
from nimo_perception.utils import Stalk

class StalkDetect:
    def __init__(self):
        # Load config
        self.loadConfig()

        if self.verbose: rospy.loginfo('Starting nimo_perception node.')

        # Check camera connection
        if not self.checkCamera(10): rospy.logwarn('Camera info not found, so camera is likely not running!')

        # Initialize variables
        if self.model_arch == "Mask_RCNN":
            self.model = Mask_RCNN.Mask_RCNN(self.model_path, self.model_threshold, self.model_device)
        else:
            rospy.logerr("Model {} not implemented".format(self.model_arch))
            raise NotImplementedError

        # Setup services
        rospy.Service('GetStalks', GetStalks, self.getStalks)
        rospy.Service('GetWidth', GetWidth, self.getWidth)

        if self.verbose: rospy.loginfo('Waiting for service calls...')

    def loadConfig(self):
        '''
        Load configuration from yaml file
        '''

        rospack = rospkg.RosPack()
        config_path = rospack.get_path('nimo_perception') + '/config/default.yaml'
        with open(config_path) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        self.verbose = config["debug"]["verbose"]
        self.visualize = config["debug"]["visualize"]
        self.save_images = config["debug"]["save_images"]

        self.camera_info_topic = config["camera"]["info_topic"]
        self.camera_image_topic = config["camera"]["image_topic"]
        self.camera_depth_topic = config["camera"]["depth_topic"]

        self.model_arch = config["model"]["model"]
        self.model_path = config["model"]["path"]
        self.model_threshold = config["model"]["score_threshold"]
        self.model_device = config["model"]["device"]

        self.minimum_mask_area = config["stalk"]["minimum_mask_area"]
        self.feature_point_offset = config["stalk"]["feature_point_offset"]

    def checkCamera(self, t=2):
        '''
        Determine whether the camera is running by checking the camera_info topic

        Parameters
            t: The timeout to wait for the camera info

        Returns
            isCameraRunning: Whether the camera_info topic has been found
        '''
        isCameraRunning = True

        try:
            camera_info = rospy.wait_for_message(self.camera_info_topic, CameraInfo, timeout=t)
            if camera_info is None:
                raise rospy.ROSException
            self.camera_height = camera_info.height
            self.camera_width = camera_info.width
        except rospy.ROSException:
            isCameraRunning = False

        return isCameraRunning

    def getStalkFeatures(self, masks, depth_image):
        '''
        Get the center points going up each stalk

        Parameters
            masks: The masks of the detected stalks
            depth_image: The current depth frame

        Returns
            stalks_features [(x, y, z), ...]: Points along the center of the stalk in the world frame
        '''

        stalks_features = []
        for mask in masks:
            # Ensure the mask has the minimum numbgetStalkFeatures(self, masks, depth_image)er of pixels
            if np.count_nonzero(mask) < self.minimum_mask_area:
                continue

            # Swap x and y in the mask
            mask = np.swapaxes(mask, 0, 1)
            nonzero = np.nonzero(mask)

            # Get the top and bottom height values of the stalk
            # NOTE: bottom_y is the top of the mask in the image using openCV indexing
            top_y, bottom_y = nonzero[1].min(), nonzero[1].max()

            # For every FEATURE_POINT_PIXEL_OFFSET pixels, get the center point
            stalk_features = []
            for y in range(top_y, bottom_y, self.feature_point_offset):
                # Find the average x index for nonzero pixels at this y value
                x_indicies = np.nonzero(mask[:, y])[0]

                # If there are no pixels, skip this value
                if len(x_indicies) > 0:
                    x_center = x_indicies.mean()

                    x = self.camera_width - x_center
                    y = self.camera_height - y
                    # TODO: Use more pixels from the depth image to get a better depth (only if they are in the mask)
                    z = depth_image[int(y), int(x)] / 1000

                    # TODO: TRANSFORM TO WORLD FRAME

                    stalk_features.append((x, y, z))

            stalks_features.append(stalk_features)

        return stalks_features

    def getStalksCallback(self, image, depth_image):
        '''
        Determine the suitable stalks within view for one frame

        Parameters
            image: The current RGB frame
            depth_image: The current depth frame
        '''

        # Convert image messaages to arrays
        depth_image = np.array(self.cv_bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough"), dtype=np.float32)
        image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding='rgb8')

        # Run detection and get feature points
        masks, output, scores = self.model.forward(image)
        stalks_features = self.getStalkFeatures(masks, depth_image)

        # Create stalk objects and add to running list
        for stalk_features, score in zip(stalks_features, scores, masks):
            stalk = Stalk.Stalk(stalk_features, score)

            if stalk.valid:
                self.stalks.append(stalk)
        
        # VISUALIZE (masked image, stalk line, grasp point)

        self.image_index += 1

    def getWidthCallback(self, image, depth_image):
        '''
        Determine the width of the closest cornstalk for the current frame

        Parameters
            image: The current RGB frame
            depth_image: The current depth frame
        '''
        self.image_index += 1

    def getStalks(self, req: GetStalksRequest) -> GetStalksResponse:
        '''
        Determine the suitable stalks within view

        Parameters
            req (GetStalkRequest): The request: 
                                   - num_frames - The number of frames to process
                                   - timeout - the timeout in seconds to wait until aborting

        Returns
            GetStalkResponse: The response:
                              - success - The success of the operation (SUCCESS / REPOSITION / ERROR)
                              - num_frames - The number of frames processed
                              - grasp_points - A list of grasp points on the cornstalks ordered from best to worst
        '''

        if self.verbose: rospy.loginfo('Received a GetStalks request for {} frames with timeout {} seconds'.format(req.num_frames, req.timeout))

        # Check camera connection
        if not self.checkCamera():
            rospy.logerr('Camera info not found, so camera is likely not running!')
            return GetStalksResponse(success='ERROR', num_frames=0)

        # Reset
        self.stalks = []
        self.image_index = 0

        # Setup callbacks
        rospy.logwarn(self.camera_depth_topic)
        rospy.logwarn(self.camera_image_topic)
        image_subscriber = message_filters.Subscriber(self.camera_image_topic, Image)
        depth_susbscriber = message_filters.Subscriber(self.camera_depth_topic, Image)
        ts = message_filters.ApproximateTimeSynchronizer([image_subscriber, depth_susbscriber], queue_size=5, slop=0.2)
        ts.registerCallback(self.getStalksCallback)

        # Wait until images have been captured
        start = rospy.get_rostime()
        while self.image_index < req.num_frames and (rospy.get_rostime() - start).to_sec() < req.timeout:
            rospy.sleep(0.1)

        # Destroy callbacks
        image_subscriber.unregister()
        depth_susbscriber.unregister()
        del ts

        # Cluster stalks + average

        # Return with list

    def getWidth(self, req: GetWidthRequest) -> GetWidthResponse:
        '''
        Determine the width of the closest cornstalk

        Parameters
            req (GetStalkRequest): The request: 
                                   - num_frames - The number of frames to process
                                   - timeout - the timeout in seconds to wait until aborting

        Returns
            GetStalkResponse: The response:
                              - success - The success of the operation (SUCCESS / REPOSITION / ERROR)
                              - num_frames - The number of frames processed
                              - width - The width of the cornstalk in mm
        '''

        if self.verbose: rospy.loginfo('Received a GetWidths request for {} frames with timeout {} seconds'.format(req.num_frames, req.timeout))

if __name__ == "__main__":
    rospy.init_node('nimo_perception')
    stalk_detect = StalkDetect()
    rospy.spin()