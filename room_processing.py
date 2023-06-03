import cv2
import numpy as np
from PIL import Image


def load_img(img_path):
    image = Image.open(img_path)
    data = np.asarray(image)
    return data


def save_image(img, img_path):
    im = Image.fromarray(img)
    im.save(img_path)


def brightness_transfer(image, wall_decorated, mask):
    """Performs brightness transfer from the original image to the new image with decorated walls.
    Convertes images from rgb to the hsv color model and some manipulations are performed with the V channel,
    which is responsible for the brightness

    Args:
        image (np.ndarray): image of an uploaded room
        wall_decorated (np.ndarray): decorated image with changed wall color or applied texture
        mask (np.ndarray): wall segmentation mask

    Returns:
        np.ndarray:decorated image with transferred brightness
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # separate channels of hsv to get v channel of original image
    h1, s1, v1 = cv2.split(hsv_image)

    hsv_image = cv2.cvtColor(wall_decorated, cv2.COLOR_BGR2HSV)
    # separate channels of hsv to get v channel of decorated image
    h2, s2, v2 = cv2.split(hsv_image)

    # finds the most frequent group of pixels corresponding to the wall mask in the original image
    # assume that the most frequent pixel corresponds to the wall area without shadows or pronounced shine
    v1_wall = v1[np.where(mask != 0)]
    hist, bin_edges = np.histogram(v1_wall.ravel(), bins=np.arange(0, 260, 5))
    result = np.argmax(hist)
    d = bin_edges[result]

    # shift every pixel corresponding to the wall on this value
    # negative values indicate shadow, positive values indicate shine
    delta = np.array(v1[np.where(mask != 0)], dtype="float32") - d

    # add these values to  V channel of the decorated image's wall pixels
    v2_wall = np.array(v2[np.where(mask != 0)], dtype="float32")
    v2_wall += delta * 1.5
    v2_wall[v2_wall > 250.0] = 250
    v2_wall[v2_wall < 0.0] = 0

    v2[np.where(mask != 0)] = np.array(v2_wall, dtype="uint8")

    # merge channels in hsv model
    hsv_image = cv2.merge([h2, s2, v2])
    out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return out
