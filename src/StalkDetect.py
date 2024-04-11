import os
import cv2
import yaml
import rospy
import rospkg
import numpy as np
import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import CameraInfo, Image
from nimo_perception.srv import *
from nimo_perception.models import Mask_RCNN
from nimo_perception.utils import stalk, utils, visualize

class StalkDetect:
    def __init__(self):
        # Load config
        self.loadConfig()

        if self.verbose: rospy.loginfo('Starting nimo_perception node.')

        # Check camera connection
        if not utils.isCameraRunning(self.camera_info_topic, 10): 
            rospy.logwarn('Camera info not found, so camera is likely not running!')
            self.camera_width = None
            self.camera_height = None
            self.camera_intrinsic = None
        else: 
            self.camera_width, self.camera_height, self.camera_intrinsic = utils.getCameraInfo(self.camera_info_topic)

        # Initialize variables
        self.cv_bridge = CvBridge()
        self.visualizer = visualize.Visualizer(self.world_frame)

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
        self.package_path = rospack.get_path('nimo_perception')
        config_path = self.package_path + '/config/default.yaml'
        with open(config_path) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        self.verbose = config["debug"]["verbose"]
        self.visualize = config["debug"]["visualize"]
        self.save_images = config["debug"]["save_images"]

        self.camera_info_topic = config["camera"]["info_topic"]
        self.camera_image_topic = config["camera"]["image_topic"]
        self.camera_depth_topic = config["camera"]["depth_topic"]
        self.world_frame = config["camera"]["world_frame"]

        self.model_arch = config["model"]["model"]
        self.model_path = self.package_path + "/weights/" + config["model"]["weights"]
        self.model_threshold = config["model"]["score_threshold"]
        self.model_device = config["model"]["device"]

        self.cluster_threshold = config["stalk"]["cluster_threshold"]

        self.config = config

    def clusterStalks(self, stalks):
        '''
        Cluster the stalks from multiple frames

        Parameters
            stalks: The list of unclustered stalks from multiple frames

        Returns
            clustered_stalks: The list of clustered stalks
        '''

        # clustering_labels = [0]
        # for position in positions[1:]:
        #     min_distance = float('inf')
        #     min_cluster = 0
        #     for i in range(len(clustering_labels)):
        #         if distance(position, positions[i]) < min_distance:
        #             min_distance = distance(position, positions[i])
        #             min_cluster = clustering_labels[i]

        #     clustering_labels.append(min_cluster if min_distance < THRESHOLD else max(clustering_labels) + 1)

        # clustering_labels = np.array(clustering_labels)

    def getStalksCallback(self, image, depth_image):
        '''
        Determine the suitable stalks within view for one frame

        Parameters
            image: The current RGB frame
            depth_image: The current depth frame
        '''

        # Reset new stalks
        new_stalks = []

        # Convert image messaages to arrays
        depth_image = np.array(self.cv_bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough"), dtype=np.float32)
        image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')

        # Run detection and get feature points
        masks, output, scores = self.model.forward(image)

        # Create stalk objects and add to running list
        for mask, score in zip(masks, scores):
            new_stalk = stalk.Stalk(mask, score, depth_image, self.camera_intrinsic, self.config)

            # Append to list if stalks are valid
            if new_stalk.valid:
                new_stalks.append(new_stalks)
        
        if self.save_images:
            features_image = self.model.visualize(image, output)
            for new_stalk in new_stalks:
                for x, y, _ in new_stalk.cam_features:
                    cv2.circle(features_image, (int(x), int(y)), 2, (255, 255, 255), -1)

            cv2.imwrite(self.package_path+"/output/FEATURES{}-{}.png".format(self.inference_index, self.image_index), features_image)

        for new_stalk in new_stalks:
            self.stalks.append(new_stalk)

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
        if not utils.isCameraRunning(self.camera_info_topic):
            rospy.logerr('Camera info not found, so camera is likely not running!')
            return GetStalksResponse(success='ERROR', num_frames=0)
        elif self.camera_width == None:
            self.camera_width, self.camera_height, self.camera_intrinsic = utils.getCameraInfo(self.camera_info_topic)

        # Reset
        self.stalks = []
        self.image_index = 0
        try:
            self.inference_index = max([int(f[len("FEATURES"):].split("-")[0]) for f in os.listdir(self.package_path+"/output")]) + 1
        except:
            self.inference_index = 0

        # Setup callbacks
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

        if len(self.stalks) == 0:
            rospy.logwarn('No valid stalks detected in any frame for this service request, requesting a REPOSITION')
            return GetStalksResponse(success='REPOSITION', num_frames=self.image_index + 1)
        
        # Cluster stalks + average

        # # Visualize
        # self.visualizer.clearMarkers()
        # self.visualizer.publishStalk(new_stalk.world_features)
        # self.visualizer.publishGraspPoint(new_stalk.grasp_point)

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