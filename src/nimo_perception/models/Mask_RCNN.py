import cv2
import numpy as np

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.catalog import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

class Mask_RCNN:
    def __init__(self, path, score_threshold, device):
        self.model_cfg = get_cfg()
        self.model_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.model_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        self.model_cfg.MODEL.WEIGHTS = path
        self.model_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # background, stalk
        self.model_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.1
        self.model_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)
        self.model_cfg.MODEL.DEVICE = 'cuda:{}'.format(device) if device >= 0 else 'cpu'

        try:
            self.model = DefaultPredictor(self.model_cfg)
        except AssertionError:
            # If the model couldn't be loaded, try to load it from the CPU
            self.model_cfg.MODEL.DEVICE = 'cpu'
            self.model = DefaultPredictor(self.model_cfg)

    def forward(self, image):
        '''
        Forward Prop for the Mask-R-CNN model

        Parameters
            image (cv.Mat/np.ndarray): the input image

        Returns
            np.ndarray: confidence scores for each prediction
            np.ndarray: (N X 4) array of bounding box corners
            np.ndarray: (N X H X W) masks for each prediction,
                where the channel 0 index corresponds to the prediction index
                and the other channels are boolean values representing whether
                that pixel is inside that prediction's mask
            dict: output object from detectron2 predictor
        '''

        outputs = self.model(image)
        scores = outputs['instances'].scores.to('cpu').numpy()
        masks = outputs['instances'].pred_masks.to('cpu').numpy()

        masks = masks.astype(np.uint8) * 255

        return masks, outputs, scores

    def visualize(self, image, output):
        '''
        Visualize the results of the model

        Parameters
            image (cv.Mat/np.ndarray): the input image
            output (dict): output from the detectron2 predictor

        Returns
            np.ndarray: visualized results
        '''

        metadata = MetadataCatalog.get('empty')
        metadata.thing_colors = [(255, 0, 0)]

        v = Visualizer(image[:, :, ::-1],
                       metadata=MetadataCatalog.get('empty'),
                       instance_mode=ColorMode.SEGMENTATION)  # remove the colors of unsegmented pixels

        v = v.draw_instance_predictions(output['instances'].to('cpu'))

        return v.get_image()[:, :, ::-1].astype(np.uint8)
