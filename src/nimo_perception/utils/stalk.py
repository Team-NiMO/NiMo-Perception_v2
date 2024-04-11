import numpy as np
import warnings
import pyransac3d as pyrsc
from nimo_perception.utils import utils

class Stalk:
    def __init__(self, stalk_features, score):

        self.features = stalk_features
        self.score = score

        # CATCH ALL ERRORS AND SET INVALID
        
        # GET LINE
        # GET WIDTH
        # GET GRASP POINT
        # GET WEIGHT

        # self.setValidity()

    def setValidity(self):
        '''
        Determine whether the stalk is valid based on score, width, and grasp point

        Returns
            valid: The validity of the stalk
        '''

        # TODO: FILTER INVALID STALKS (SCORE, WIDTH, GRASP POINT)
        self.valid = self.valid