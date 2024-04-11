import numpy as np
import warnings
import pyransac3d as pyrsc
from nimo_perception.utils import utils

# TODO: Pass in parameters
# TODO: Fix features getting swapped around

class Stalk:
    def __init__(self, mask, score, depth_image, camera_intrinsic, config):
        self.valid = True
        self.score = score
        self.camera_height, self.camera_width = depth_image.shape
        self.camera_intrinsic = camera_intrinsic

        self.loadConfig(config)

        # Get stalk features in camera frame
        self.cam_features = self.getFeatures(mask, depth_image)

        # Get stalk width
        self.width = self.getWidth(self.cam_features, mask, depth_image)

        # Transform features to world frame
        self.world_features = self.transformFeatures(self.cam_features)

        # Get stalk line
        self.stalk_line = self.getLine(self.world_features)
        
        # Get grasp point
        self.grasp_point = self.getGrasp(self.world_features)
        
        # GET WEIGHT

        self.valid = self.isValid(self.valid)

    def loadConfig(self, config):
        '''
        Load config specified by yaml file

        Parameters
            config: The stalk config from the cofniguration file
        '''
        self.minimum_mask_area = config["stalk"]["minimum_mask_area"]
        self.feature_point_offset = config["stalk"]["feature_point_offset"]
        self.optimal_grasp_height = config["stalk"]["optimal_grasp_height"]

        self.camera_frame = config["camera"]["camera_frame"]
        self.world_frame = config["camera"]["world_frame"]

    def getFeatures(self, mask, depth_image):
        '''
        Get the center points going up each stalk in the camera frame

        Parameters
            mask: The mask of the detected stalk
            depth_image: The current depth frame

        Returns
            stalk_features [(x, y, z), ...]: Points along the center of the stalk in the camera frame
        '''

        # Ensure the mask has the minimum numbgetStalkFeatures(self, masks, depth_image)er of pixels
        if np.count_nonzero(mask) < self.minimum_mask_area:
            self.valid = False
            return

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

                # TODO: Use more pixels from the depth image to get a better depth (only if they are in the mask)
                z = depth_image[int(y) - 1, int(x_center)] / 1000
                # x = self.camera_width - x_center
                # y = self.camera_height - y
                x = x_center
                y = y

                stalk_features.append((x, y, z))

        return stalk_features

    def getWidth(self, features, mask, depth_image):
        '''
        Get the width of the stalk

        Parameters
            features: The stalk features in the world frame
            mask: The mask of the detected stalk
            depth_image: The current depth frame

        Returns
            width: The width of the stalk in mm
        '''

        if len(features) == 0:
            self.valid = False
            return 0

        try:
            slope, _, _ = utils.ransac_2d(features)
        except:
            self.valid = False
            return 0

        perp_slope = -1 / slope

        nonzero = np.nonzero(mask)
        x0 = nonzero[1].min() # MIN X
        y1 = nonzero[0].min() # MIN Y
        x2 = nonzero[1].max() # MAX X
        x3, y3 = (nonzero[1][np.argmax(nonzero[0])], nonzero[0].max()) # MAX Y AND CORRESPONDING X

        widths = []
        px_widths = []
        # Loop bottom to top of mask
        max_y = int(y1 - abs((x0-x2) * perp_slope))
        min_y = int(y3 + abs((x0-x2) * perp_slope))
        for i in range(min_y, max_y, -1): # MOTIVATE Y LIMITS BY NECESSARY X VALUES
            # Loop across every line
            count = 0
            depths = []
            for j in range(x0-x2, x2-x0):
                test_x, test_y = (int(j + x3), int(i + perp_slope * j))
                try:
                    if mask[test_y, test_x]:
                        mask[test_y, test_x] = 125
                        count += 1
                        if depth_image[test_y, test_x] != 0:
                            depths.append(depth_image[test_y, test_x])
                except:
                    pass
            if count > 0 and len(depths) > 0:
                widths.append(count * np.median(depths) * 0.0036) # NOTE: MAGIC NUM FIX LATER
                px_widths.append(count)

        return np.median(widths)
    
    def transformFeatures(self, cam_features):
        '''
        Transform features from camera frame to the world frame

        Parameters
            cam_features: The stalk features in the camera frame

        Returns
            world_features: The stalk features in the world frame
        '''
        
        world_features = []
        for c_x, c_y, c_z in cam_features:
           x, y, z = utils.transformCam2World((self.camera_width - c_x, self.camera_height - c_y, c_z), self.camera_intrinsic, self.camera_frame, self.world_frame)
           world_features.append((x, y, z))

        return world_features

    def getLine(self, features):
        '''
        Perform RANSAC line detection on the stalk features

        Parameters
            features: The stalk features in the world frame

        Returns
            best_line: The best line found
        '''
        points = np.array(features)

        # Catch the RuntimeWarning that pyransac3d throws when it fails to find a line
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            line = pyrsc.Line().fit(points, thresh=0.025, maxIteration=1000)
            if len(w) > 0 and not issubclass(w[-1].category, RuntimeWarning):
                self.valid &= False

        return line
    
    def getGrasp(self, stalk_features):
        '''
        Get the grasp point for the cornstalk

        Parameters
            stalk_features: The stalk features in the world frame

        Returns
            grasp_point: The grasp point in the world frame
        '''

        # TODO: FIX GRASP POINT (MAKE ON THE STALK)

        heights = np.array([z for _, _, z in stalk_features])
        x, y, z = stalk_features[np.argmin(heights)]
        return (x, y, z)

    def isValid(self, valid):
        '''
        Determine whether the stalk is valid based on score, width, and grasp point

        Returns
            valid: The validity of the stalk
        '''

        # TODO: FILTER INVALID STALKS (SCORE, WIDTH, GRASP POINT)
        if self.score < 0.9:
            valid = False

        return valid