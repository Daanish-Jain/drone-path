
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

import torch
import cv2


_predictor = None
_metadata = None


def load_model():

    global _predictor, _metadata

    if _predictor is None:

        cfg = get_cfg()

        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
            )
        )

        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
        )

        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        _predictor = DefaultPredictor(cfg)

        _metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    return _predictor, _metadata


def run_panoptic_segmentation(image):

    predictor, metadata = load_model()

    outputs = predictor(image)

    panoptic_seg, segments_info = outputs["panoptic_seg"]

    return panoptic_seg.to("cpu"), segments_info, metadata


def visualize_panoptic(image, panoptic_seg, segments_info, metadata):

    visualizer = Visualizer(
        image[:, :, ::-1],
        metadata,
        scale=1.0
    )

    out = visualizer.draw_panoptic_seg(
        panoptic_seg,
        segments_info
    )

    result = out.get_image()[:, :, ::-1]

    return result
