from .models.models import SegmentationModule, build_encoder, build_decoder
from .src.eval import segment_image
from .utils.constants import DEVICE
import os


def build_model():
    """Initializing pretrained model

    Returns:
        SegmentationModule: pretrained model for wall segmentation
    """
    weights_decoder = os.path.join(
        "wall_segmentation", "weights", "wall_decoder_epoch_20.pth"
    )
    weights_encoder = os.path.join(
        "wall_segmentation", "weights", "wall_encoder_epoch_20.pth"
    )

    net_encoder = build_encoder(weights_encoder)
    net_decoder = build_decoder(weights_decoder)

    segmentation_model = SegmentationModule(net_encoder, net_decoder)
    segmentation_model = segmentation_model.to(DEVICE).eval()

    return segmentation_model


def wall_segmenting(model, path_image):
    """Predict wall on input image and return binary segmentation mask

    Args:
        model (SegmentationModule): pretrained model
        path_image (str): path to image of a room

    Returns:
        np.ndarray: segmentation binary mask
    """
    output = segment_image(model, path_image)
    output = output ^ 1
    return output
