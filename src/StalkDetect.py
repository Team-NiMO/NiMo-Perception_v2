import os
import cv2
import yaml
import rospy
import rospkg
import tf2_ros
import datetime
import numpy as np
import message_filters
from cv_bridge import CvBridge

from geometry_msgs.msg import Point
from sensor_msgs.msg import Image

from nimo_perception.srv import *
from nimo_perception.msg import *
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
            self.camera_intrinsic = None
        else: 
            self.camera_intrinsic = utils.getCameraInfo(self.camera_info_topic)

        # Check world transform
        try:
            utils.getCam2WorldTransform(self.camera_frame, self.world_frame)
        except:
            rospy.logwarn('Camera to world transform not found')

        # Initialize variables
        self.cv_bridge = CvBridge()
        self.visualizer = visualize.Visualizer(self.world_frame)

        if self.model_arch == "Mask_RCNN":
            try:
                self.model = Mask_RCNN.Mask_RCNN(self.model_path, self.model_threshold, self.model_device)
            except RuntimeError as e:
                rospy.logerr("Device {} not available, set device to -1 in default.yaml to use CPU".format(self.model_device))
                raise Exception("Device {} not available, set device to -1 in default.yaml to use CPU".format(self.model_device))
        else:
            rospy.logerr("Model {} not implemented".format(self.model_arch))
            raise NotImplementedError

        # Setup services
        rospy.Service('GetStalks', GetStalks, self.getStalks)
        rospy.Service('GetWidth', GetWidth, self.getWidth)
        rospy.Service('GetRefinedGrasp', GetRefinedGrasp, self.getRefinedGrasp)

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
        self.camera_frame = config["camera"]["camera_frame"]
        self.camera_brightness = config["camera"]["brightness"]

        self.model_arch = config["model"]["model"]
        self.model_path = self.package_path + "/weights/" + config["model"]["weights"]
        self.model_threshold = config["model"]["score_threshold"]
        self.model_device = config["model"]["device"]

        self.cluster_threshold = config["stalk"]["cluster_threshold"]
        self.x_diff_clearance = config["stalk"]["x_diff_clearance"]

        self.minimum_x = config["stalk"]["min_x"]
        self.maximum_x = config["stalk"]["max_x"]

        self.config = config

    def clusterStalks(self, stalks):
        '''
        Cluster the stalks from multiple frames

        Parameters
            stalks: The list of unclustered stalks from multiple frames

        Returns
            sorted_grasp_points: The list of clustered grasp_points sorted from highest to lowest weight
            sorted_weights: The list of clustered weights sorted from highest to lowest
            sorted_widths: The list of clustered widths sorted from highest to lowest weight
        '''

        # Determine clusters
        clustering_labels = [0]
        for stalk in stalks[1:]:
            min_distance = float('inf')
            min_cluster = 0
            for i in range(len(clustering_labels)):
                # NOTE: Not accounting for z, because the same stalk may have the same x & y, but different z
                dist = np.linalg.norm(np.array(stalk.grasp_point[:2]) - np.array(stalks[i].grasp_point[:2]))
                if dist < min_distance:
                    min_distance = dist
                    min_cluster = clustering_labels[i]

            clustering_labels.append(min_cluster if min_distance < self.cluster_threshold else max(clustering_labels) + 1)

        clustering_labels = np.array(clustering_labels)

        # Combine clusters
        clustered_grasp_points = []
        clustered_weights = []
        clustered_widths = []
        for label in np.unique(clustering_labels):
            grasp_points = [stalk.grasp_point for stalk in np.array(stalks)[np.nonzero(clustering_labels == label)]]
            if self.for_visual_servoing:
                weights = [stalk.dist_to_initial for stalk in np.array(stalks)[np.nonzero(clustering_labels == label)]]
            else:
                weights = [stalk.weight for stalk in np.array(stalks)[np.nonzero(clustering_labels == label)]]
            widths = [stalk.width for stalk in np.array(stalks)[np.nonzero(clustering_labels == label)]]

            # NOTE: Not robust to low negative outliers (could detect false low grasp point)
            min_z_idx = np.argmin([z for _, _, z in grasp_points])
            grasp_avg = Point(x=grasp_points[min_z_idx][0],
                              y=grasp_points[min_z_idx][1],
                              z=grasp_points[min_z_idx][2])
            
            weight_avg = np.mean(weights)

            width_avg = np.mean(widths)

            clustered_grasp_points.append(grasp_avg)
            clustered_weights.append(weight_avg)
            clustered_widths.append(width_avg)

        # End Effector Width + Clearance on both sides
        X_DIFF_THRESH = 0.08 + 0.05 + self.x_diff_clearance

        clustered_weights = [x for _, x in sorted(zip(clustered_grasp_points, clustered_weights), key=lambda pair: pair[0].x, reverse=False)]
        clustered_widths = [x for _, x in sorted(zip(clustered_grasp_points, clustered_widths), key=lambda pair: pair[0].x, reverse=False)]
        clustered_grasp_points = sorted(clustered_grasp_points, key = lambda point: point.x, reverse = False)

        clustered_grasp_points_in_x = [point.x for point in clustered_grasp_points]
        diff_in_x = np.diff(clustered_grasp_points_in_x)
        diff_mask = np.concatenate((np.array([True]), diff_in_x >= X_DIFF_THRESH)).astype(bool)

        rospy.logwarn(diff_mask)

        if not diff_mask.all():
            if self.verbose: rospy.loginfo('Stalks too close together, some stalks removed')

        clustered_weights = [x for y, x in zip(diff_mask, clustered_weights) if y]
        clustered_widths = [x for y, x in zip(diff_mask, clustered_widths) if y]
        clustered_grasp_points = [x for y, x in zip(diff_mask, clustered_grasp_points) if y]

        temp_clustered_widths = []
        temp_clustered_weights = []
        temp_clustered_grasp_points = []
        for i, clustered_grasp_point in enumerate(clustered_grasp_points):
            if abs(clustered_grasp_point.x) > self.minimum_x and abs(clustered_grasp_point.x) < self.maximum_x:
                temp_clustered_widths.append(clustered_widths[i])
                temp_clustered_weights.append(clustered_weights[i])
                temp_clustered_grasp_points.append(clustered_grasp_points[i])

        clustered_widths = temp_clustered_widths
        clustered_weights = temp_clustered_weights
        clustered_grasp_points = temp_clustered_grasp_points

        # Sort representative stalks by weight
        if self.for_visual_servoing: # Sort by smallest distance to original grasp point to largest distance to original grasp point
            sorted_grasp_points = [x for _, x in sorted(zip(clustered_weights, clustered_grasp_points), key=lambda pair: pair[0], reverse=False)]
            sorted_widths = [x for _, x in sorted(zip(clustered_weights, clustered_widths), key=lambda pair: pair[0], reverse=False)]
            sorted_weights = sorted(clustered_weights, reverse=False)
        else: # Sort by largest weight to smallest weight
            sorted_grasp_points = [x for _, x in sorted(zip(clustered_weights, clustered_grasp_points), key=lambda pair: pair[0], reverse=True)]
            sorted_widths = [x for _, x in sorted(zip(clustered_weights, clustered_widths), key=lambda pair: pair[0], reverse=True)]
            sorted_weights = sorted(clustered_weights, reverse=True)

        return sorted_grasp_points, sorted_weights, sorted_widths

    def getStalksCallback(self, image, depth_image):
        '''
        Determine the suitable stalks within view for one frame

        Parameters
            image: The current RGB frame
            depth_image: The current depth frame
        '''

        # Reset new stalks
        new_stalks = []

        if self.verbose: rospy.loginfo("Capturing Image {}".format(self.image_index))

        # Convert image messaages to arrays
        depth_image = np.array(self.cv_bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough"), dtype=np.float32)
        image = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')

        # Decrease brightness
        image = image * self.camera_brightness

        # Run detection and get feature points
        masks, output, scores = self.model.forward(image)

        # Create stalk objects and add to running list
        for mask, score in zip(masks, scores):
            if self.for_visual_servoing:
                new_stalk = stalk.Stalk(mask, score, depth_image, self.camera_intrinsic, self.config, self.initial_grasp_point)
                if new_stalk.valid != 0:
                    new_stalks.append(new_stalk)
            else:
                new_stalk = stalk.Stalk(mask, score, depth_image, self.camera_intrinsic, self.config)
                # Append to list if stalks are valid
                if new_stalk.valid != 0:
                    new_stalks.append(new_stalk)

        if self.visualize:
            for new_stalk in new_stalks:
                self.visualizer.publishStalk(new_stalk.world_features)
        
        if self.save_images:
            features_image = self.model.visualize(image, output)
            for new_stalk in new_stalks:
                for x, y, _ in new_stalk.cam_features:
                    cv2.circle(features_image, (int(x), int(y)), 2, (255, 255, 255), -1)

            cv2.imwrite(self.package_path+"/output/FEATURES{}-{}.png".format(self.inference_index, self.image_index), features_image)
            
            time_str = datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
            cv2.imwrite(self.package_path+"/data/IMAGE_{}.png".format(time_str), image)
            np.save(self.package_path+"/data/DEPTH_{}.png".format(time_str), depth_image)

        for new_stalk in new_stalks:
            self.stalks.append(new_stalk)

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
        elif self.camera_intrinsic is None:
            self.camera_intrinsic = utils.getCameraInfo(self.camera_info_topic)

        # Check world transform
        try:
            utils.getCam2WorldTransform(self.camera_frame, self.world_frame)
        except:
            rospy.logerr('Camera to world transform not found')
            return GetStalksResponse(success='ERROR', num_frames=0)

        # Reset
        self.stalks = []
        self.image_index = 0
        try:
            self.inference_index = max([int(f[len("FEATURES"):].split("-")[0]) for f in os.listdir(self.package_path+"/output")]) + 1
        except:
            self.inference_index = 0
        self.for_visual_servoing = False

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
        
        # Cluster stalks + sort
        clustered_grasp_points, clustered_weights, _ = self.clusterStalks(self.stalks)

        # # FIX STALKS TO CLOSE TO EACH OTHER
        # n = len(clustered_grasp_points)
        # delete_ind = []
        # for i in range(n):
        #     for j in range(i+1,n):
        #         point1 = clustered_grasp_points[i]
        #         point2 = clustered_grasp_points[j]
        #         if (point1.x - point2.x) < 0.12:
        #             delete_ind.append(i)
        #             delete_ind.append(j)
        
        # delete_ind = list(set(delete_ind))
        # for i in range(len(delete_ind), 0, -1):
        #     del clustered_grasp_points[delete_ind[i]]
        #     del clustered_weights[delete_ind[i]]

        grasp_msgs = []
        for grasp_point, weight in zip(clustered_grasp_points, clustered_weights):
            grasp_msgs.append(GraspPoint(position=grasp_point, weight=weight))
            self.visualizer.publishGraspPoint((grasp_point.x, grasp_point.y, grasp_point.z))

        # Return with list
        return GetStalksResponse(success="DONE", grasp_points=grasp_msgs, num_frames=self.image_index+1)

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

        # Check camera connection
        if not utils.isCameraRunning(self.camera_info_topic):
            rospy.logerr('Camera info not found, so camera is likely not running!')
            return GetWidthResponse(success='ERROR', num_frames=0)
        elif self.camera_intrinsic is None:
            self.camera_intrinsic = utils.getCameraInfo(self.camera_info_topic)

        # Check world transform
        try:
            utils.getCam2WorldTransform(self.camera_frame, self.world_frame)
        except:
            rospy.logerr('Camera to world transform not found')
            return GetStalksResponse(success='ERROR', num_frames=0)

        # Reset
        self.stalks = []
        self.image_index = 0
        self.visualizer.clearMarkers()
        try:
            self.inference_index = max([int(f[len("FEATURES"):].split("-")[0]) for f in os.listdir(self.package_path+"/output")]) + 1
        except:
            self.inference_index = 0
        self.for_visual_servoing = False

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
            rospy.logwarn('No valid stalks detected in any frame for this service request')
            return GetWidthResponse(success='ERROR', num_frames=self.image_index + 1)

        # Cluster stalks + sort
        clustered_grasp_points, _, clustered_widths = self.clusterStalks(self.stalks)

        # Get location of the camera
        tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tfBuffer)

        world_to_cam = tfBuffer.lookup_transform(self.world_frame, self.camera_frame, rospy.Time(0), rospy.Duration.from_sec(0.5)).transform.translation
        camera_location = (world_to_cam.x, world_to_cam.y, world_to_cam.z)

        # Find closest stalk to camera
        dists = []
        for grasp_point in clustered_grasp_points:
            point = (grasp_point.x, grasp_point.y, grasp_point.z)
            dists.append(np.linalg.norm(np.array(camera_location) - np.array(point)))

        # If closest stalk is within threshold, return width
        try:
            min_dist = min(dists)
        except:
            min_dist = np.inf
            
        if min_dist < 0.25:
            return GetWidthResponse(success="DONE", width=clustered_widths[np.argmin(dists)], num_frames=self.image_index+1)
        else:
            rospy.logwarn('Nearest stalk is not detected')
            return GetWidthResponse(success='ERROR', num_frames=self.image_index + 1)

    def getRefinedGrasp(self, req: GetRefinedGraspRequest) -> GetRefinedGraspResponse:
        '''
        Determine the width of the closest cornstalk

        Parameters
            req (GetRefinedGraspRequest): The request: 
                                   - num_frames - The number of frames to process
                                   - timeout - the timeout in seconds to wait until aborting
                                   - initial_grasp_point - Point (geometry_msgs), initial grasp point when IDing best cornstalk

        Returns
            GetRefinedGraspResponse: The response:
                              - success - The success of the operation (SUCCESS / REPOSITION / ERROR)
                              - num_frames - The number of frames processed
                              - refined_grasp_point - Point (geometry_msgs), new/best grasp point from closer view of cornstalk
        '''

        if self.verbose: rospy.loginfo('Received a GetRefinedGrasp request for {} frames with timeout {} seconds'.format(req.num_frames, req.timeout))

        # Check camera connection
        if not utils.isCameraRunning(self.camera_info_topic):
            rospy.logerr('Camera info not found, so camera is likely not running!')
            return GetRefinedGraspResponse(success='ERROR', num_frames=0, refined_grasp_point=Point(0., 0., 0.))
        elif self.camera_intrinsic is None:
            self.camera_intrinsic = utils.getCameraInfo(self.camera_info_topic)

        # Check world transform
        try:
            utils.getCam2WorldTransform(self.camera_frame, self.world_frame)
        except:
            rospy.logerr('Camera to world transform not found')
            return GetRefinedGraspResponse(success='ERROR', num_frames=0, refined_grasp_point=Point(0., 0., 0.))

        # Reset
        self.stalks = []
        self.image_index = 0
        self.visualizer.clearMarkers()
        try:
            self.inference_index = max([int(f[len("FEATURES"):].split("-")[0]) for f in os.listdir(self.package_path+"/output")]) + 1
        except:
            self.inference_index = 0
        self.for_visual_servoing = True
        self.initial_grasp_point = req.initial_grasp_point

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
            rospy.logwarn('No valid stalks detected in any frame for this service request')
            return GetRefinedGraspResponse(success='ERROR', num_frames=self.image_index + 1, refined_grasp_point=Point(0., 0., 0.))

        # Cluster stalks + sort
        clustered_grasp_points, _, _ = self.clusterStalks(self.stalks)

        try:
            return GetRefinedGraspResponse(success="DONE", num_frames=self.image_index+1, refined_grasp_point = clustered_grasp_points[0])
        except:
            rospy.logwarn('No valid stalks detected in any frame for this service request')
            return GetRefinedGraspResponse(success='ERROR', num_frames=self.image_index + 1, refined_grasp_point=Point(0., 0., 0.))

if __name__ == "__main__":
    rospy.init_node('nimo_perception')
    stalk_detect = StalkDetect()
    rospy.spin()