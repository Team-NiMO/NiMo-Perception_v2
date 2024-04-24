import numpy as np
import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

class Visualizer:
    def __init__(self, marker_frame):
        # Define templates for stalk line and grasp point
        self.defineTemplates(marker_frame)

        # Initialize publishers
        self.stalk_publisher = rospy.Publisher("stalk_line", Marker, queue_size=10)
        self.grasp_publisher = rospy.Publisher("grasp_point", Marker, queue_size=10)

        self.id = 0

    def clearMarkers(self):
        '''
        Clear all of the markers currently displayed
        '''
        self.stalk_publisher.publish(self.remove_markers)
        self.grasp_publisher.publish(self.remove_markers)
        self.id = 0

    def defineTemplates(self, marker_frame):
        '''
        Define the templates for stalk line and grasp point

        Parameters
            marker_frame: The frame to which the markers should be published
        '''

        # Stalk marker
        self.stalk_marker = Marker()
        self.stalk_marker.header.frame_id = marker_frame
        self.stalk_marker.ns = 'points_to_line'
        self.stalk_marker.type = Marker.LINE_STRIP
        self.stalk_marker.action = Marker.ADD
        self.stalk_marker.pose.orientation.w = 1.0
        self.stalk_marker.scale.x = 0.01
        self.stalk_marker.color.g = 1.0
        self.stalk_marker.color.a = 1.0

        # Grasp marker
        self.grasp_marker = Marker()
        self.grasp_marker.header.frame_id = marker_frame
        self.grasp_marker.type = 2
        self.grasp_marker.pose.orientation.w = 1.0
        self.grasp_marker.scale.x = 0.015
        self.grasp_marker.scale.y = 0.015
        self.grasp_marker.scale.z = 0.015
        self.grasp_marker.color.r = 1.0
        self.grasp_marker.color.g = 0.08
        self.grasp_marker.color.b = 0.6
        self.grasp_marker.color.a = 1.0

        # Clear markers
        self.remove_markers = Marker()
        self.remove_markers.id = 0
        self.remove_markers.action = Marker.DELETEALL

    def publishStalk(self, stalk_features):
        '''
        Create a stalk line marker based on stalk features

        Parameters
            stalk_features: The stalk features to turn into a line

        Returns
            marker: The line marker
        '''
        
        heights = np.array([z for _, _, z in stalk_features])
        min_p = stalk_features[np.argmin(heights)]
        max_p = stalk_features[np.argmax(heights)]

        bottom_point = Point(x=min_p[0], y=min_p[1], z=min_p[2])
        top_point = Point(x=max_p[0], y=max_p[1], z=max_p[2])

        marker = self.stalk_marker
        marker.header.stamp = rospy.Time.now()
        marker.id = self.id
        marker.points = [top_point, bottom_point]

        self.id += 1
        
        self.stalk_publisher.publish(marker)

    def publishGraspPoint(self, grasp_point):
        '''
        Create a grasp point marker based on the grasp point

        Parameters
            grasp_point: The grasp point to turn into a marker

        Returns
            marker: The grasp point marker
        '''

        x, y, z = grasp_point
        marker = self.grasp_marker
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.id = self.id

        self.id += 1

        self.grasp_publisher.publish(marker)