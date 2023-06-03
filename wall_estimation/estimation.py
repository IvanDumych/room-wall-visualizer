import cv2
import numpy as np
import torch
from .datasets import sequence
from .trainer import core
import os

torch.backends.cudnn.benchmark = True


class Predictor:
    def __init__(self, weight_path):
        self.model = core.LayoutSeg.load_from_checkpoint(
            weight_path, backbone="resnet101", map_location="cpu"
        )
        self.model.freeze()

    @torch.no_grad()
    def feed(self, image: torch.Tensor, alpha=0.4) -> np.ndarray:
        _, outputs = self.model(image.unsqueeze(0))
        label = core.label_as_rgb_visual(outputs.cpu()).squeeze(0)
        return label.permute(1, 2, 0).numpy()


# Load predictor
predictor = Predictor(
    weight_path=os.path.join("wall_estimation", "weight", "model_retrained.ckpt")
)


def wall_estimation(path, image_size=320):
    """Perform room segmentation to get position of separate walls

    Args:
        path (str): path to image of a room
        image_size (int, optional): size for scaling image. Defaults to 320.

    Returns:
        np.ndarray: segmentation result, where each pixel equel to one of 5 classes
        (left wall, central wall, right wall, ceil and floor)
    """
    images = sequence.ImageFolder(image_size, path)
    path = os.path.abspath(path)

    image, shape, _ = list(images)[0]

    output = predictor.feed(image)
    output = cv2.resize(output, shape)

    result = output[..., ::-1].astype(np.uint8) * 255

    return result
