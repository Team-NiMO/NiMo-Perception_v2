import numpy as np
import warnings
import pyransac3d as pyrsc

# TODO: Pass in parameters
# TODO: MOVE GET FEATURES TO STALK

class Stalk:
    def __init__(self, mask, score, depth_image, config):
        self.valid = True
        self.score = score
        self.camera_height, self.camera_width = depth_image.shape

        self.loadConfig(config)

        # Get stalk features in camera frame
        self.cam_features = self.getFeatures(mask, depth_image)

        # Get stalk width
        self.width = self.getWidth(self.features, mask, depth_image)

        # TRANSFORM FEATURES TO WORLD FRAME
        # x, y, z = utils.transformCam2World((x, y, z), self.camera_intrinsic, self.camera_frame, self.world_frame)
        # # Get stalk line
        # self.stalk_line = self.getLine(self.features)
        # GET GRASP POINT
        
        # GET WEIGHT

        self.setValidity()

    def loadConfig(self, config):
        '''
        Load config specified by yaml file

        Parameters
            config: The stalk config from the cofniguration file
        '''
        self.minimum_mask_area = config["minimum_mask_area"]
        self.feature_point_offset = config["feature_point_offset"]

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
                x = self.camera_width - x_center
                y = self.camera_height - y

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
        pass

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
        return max([z for _, _, z in stalk_features]) + OPTIMAL_STALK_HEIGHT

    def setValidity(self):
        '''
        Determine whether the stalk is valid based on score, width, and grasp point

        Returns
            valid: The validity of the stalk
        '''

        # TODO: FILTER INVALID STALKS (SCORE, WIDTH, GRASP POINT)
        self.valid = self.valid