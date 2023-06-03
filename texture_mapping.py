import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import math


def load_img(img_path):
    image = Image.open(img_path)
    data = np.asarray(image)
    return data


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(40, 30))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title(), color="black")
        plt.imshow(image)
    plt.show()


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def get_wall_corners(image):
    """Computes the coordinates of walls according to the computed segmentation mask

    Args:
        image (np.ndarray): wall layout estimation segmentation mask 

    Returns:
        List[List[Tuple[int, int]]]: coordinates of the corresponding walls
    """
    # need bgr format
    image = image[..., ::-1]

    rgb_unique = set(
        tuple(rgb) for rgb in image.reshape(image.shape[0] * image.shape[1], 3)
    )

    result = []

    # needed only 3 classes of segmentation(left wall, right wall, central wall)
    # each color indicates wall class
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for color in colors:
        if color not in rgb_unique:
            continue

        mask = np.all(image == color, axis=-1)
        mask = mask.astype(np.uint8)
        img = np.copy(image)
        img[np.where(mask != 1)] = [0, 0, 0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, tresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

        # find contours of a wall
        contours, _ = cv2.findContours(
            tresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(contours, key=lambda x: cv2.contourArea(x))

        # approximate polygon for less vertices
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.005 * perimeter, True)

        # check if area of wall side is satisfactory
        area = cv2.contourArea(approx)

        if area < (image.shape[0] * image.shape[1]) / 20:
            continue

        # find 4 corners
        approx = [tuple(point[0]) for point in approx]
        points = countour_rect_corners(approx)

        points.sort()

        if points[0][1] > points[1][1]:
            points[0], points[1] = points[1], points[0]

        if points[2][1] > points[3][1]:
            points[2], points[3] = points[3], points[2]

        result.append(points)

    # changes the left coordinates of the wall according to the previous one for a smoother transition
    for i in range(1, len(result)):
        right_top_point = result[i - 1][2]
        right_bottom_points = result[i - 1][3]
        result[i][0] = (right_top_point[0], result[i][0][1])
        result[i][1] = right_bottom_points

    for points in result:
        points[3], points[2] = points[2], points[3]

    return result


def countour_rect_corners(approx):
    """Approximate vertices of polygon to 4 points indicating the edges of the wall

    Args:
        approx (List[Tuple[int, int]]): vertices of polygon of the wall

    Returns:
        List[Tuple[int, int]]: vertices of the wall
    """
    points = []
    max_x = max(approx, key=lambda p: p[0])[0]
    min_x = min(approx, key=lambda p: p[0])[0]
    max_y = max(approx, key=lambda p: p[1])[1]
    min_y = min(approx, key=lambda p: p[1])[1]

    width = max_x - min_x
    height = max_y - min_y

    # Calculate some threshold of height and width values for finding vertices
    tresh_left_x = width / 6 + min_x
    tresh_y = (height * 3) / 5 + min_y

    # find top left point
    filtered = filter(lambda p: p[0] < tresh_left_x and p[1] < tresh_y, approx)
    filtered = list(filtered)

    top_left_point = min(filtered, key=lambda p: (p[1], p[0]))

    # find top right point
    tresh_right_x = (width * 4) / 5 + min_x
    filtered = filter(lambda p: p[0] > tresh_right_x and p[1] < tresh_y, approx)
    filtered = list(filtered)

    top_right_point = min(filtered, key=lambda p: (p[1], -p[0]))

    # Approxe if coordinates y is above
    # In this case, the entire wall is not visible in the image,and the top vertex is somewhere higher with a negative axis value
    # It is necessary to correctly find a vertex with negative coordinates on one of the edges
    if top_left_point[1] > 25 and top_right_point[1] < 10:
        top_right_point = find_approx_top(approx, top_left_point, top_right_point, 1)
    elif top_left_point[1] < 10 and top_right_point[1] > 25:
        top_left_point = find_approx_top(approx, top_right_point, top_left_point, -1)

    # find bottom left point
    filtered = filter(lambda p: p[0] < tresh_left_x and p[1] > tresh_y, approx)
    filtered = list(filtered)

    bottom_left_point = min(filtered, key=lambda p: p[0])

    # find bottom right point
    filtered = filter(lambda p: p[0] > tresh_right_x and p[1] > tresh_y, approx)
    filtered = list(filtered)

    bottom_right_point = min(filtered, key=lambda p: -p[0])

    points.append(top_left_point)
    points.append(top_right_point)
    points.append((points[0][0], bottom_left_point[1]))
    points.append((points[1][0], bottom_right_point[1]))

    return points


def find_approx_top(approx, top_left_point, top_right_point, sign=1):
    """
    Calculates the new y-axis coordinates for the top(left or right)point
    if one of the upper corners of the wall is not visible in the image.
    Basic trigonometric dependencies are used to find new coordinates of a point
    that would estimate the approximate location of the top of the wall

    Args:
        approx (List[Tuple[int, int]]): vertices of polygon of the wall
        top_left_point (Tuple[int, int]): uppermost point of the wall
        top_right_point (Tuple[int, int]): uppermost point of the wall
        sign (int, optional): Indicates is it top left point or top right. Defaults to 1 for right point

    Returns:
        Tuple[int, int]: new top point that is above on image
    """

    def side_wall_zero_y(points):
        """Finds upper left or upper right point which
        indicates the edge of the wall on which the wall image is cropped.
        It forms with the extreme top(left or right) some certain vector
        that indicates the direction of the edges of the wall
        """
        points_zero_y = list(filter(lambda p: p[1] < 10, points))
        point = min(points_zero_y, key=lambda p: (sign * p[0]))
        return point

    second_top_right_point = side_wall_zero_y(approx)

    if second_top_right_point != top_right_point:
        a, b, c = (
            (-top_left_point[0] - sign, top_left_point[1]),
            (-top_left_point[0], top_left_point[1]),
            (-second_top_right_point[0], second_top_right_point[1]),
        )

        angle = getAngle(a, b, c)
        if angle > 180:
            angle = 180 - angle

        d = abs(top_right_point[0] - top_left_point[0])

        stroke_c = math.tan(math.radians(angle)) * d

        top_right_point = (
            top_right_point[0],
            int(top_right_point[1] + (top_left_point[1] - stroke_c)),
        )

    return top_right_point


def getAngle(a, b, c):
    """Compute angle between 3 points"""
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return ang + 360 if ang < 0 else ang


def map_texture(texture, image, dsts, mask):
    """Performs a perspective transformation of the texture on the walls of the room according to the
    computed coordinates

    Args:
        texture (np.ndarray): uploaded texture
        image (np.ndarray): image of an uploaded room
        dsts (np.ndarray): array of walls coordinates where 4 points corresponding to the edges of the wall
        mask (np.ndarray): wall segmentation mask

    Returns:
        np.ndarray: textured room image
    """
    img = image.copy()

    # change slightly pixel rgb value for correct further texture applying
    texture[np.all(texture == [0, 0, 0], axis=-1)] = (1, 1, 1)

    # dst indicates coordinates of separate wall
    for dst in dsts:
        dst = dst.tolist()

        height = texture.shape[0]
        width = texture.shape[1]

        src = np.float32([[0, 0], [0, height], [width, 0], [width, height]])

        # swap for appropriate src order
        dst[3], dst[2] = dst[2], dst[3]

        dst = np.float32(dst)

        # perspective transform
        M = cv2.getPerspectiveTransform(src.astype("float32"), dst.astype("float32"))
        warped = cv2.warpPerspective(texture, M, image.shape[:2][::-1])

        # mask for texture
        mask_texture = np.zeros(image.shape, dtype=np.uint8)

        # rgb value (0, 0, 0) indicates area that does not correspond to the texture
        mask_texture[np.all(warped == (0, 0, 0), axis=-1)] = (255, 255, 255)

        masked_image = cv2.bitwise_and(img, mask_texture)

        # change the pixels of the corresponding wall to the transformed texture
        img = cv2.bitwise_or(warped, masked_image)

    final = image.copy()

    # change wall pixels on transformed texture pixels according to the mask
    final[np.where(mask != 0)] = img[np.where(mask != 0)]

    return final


def load_texture(path, n=5, m=5):
    """Texture extension along each axis for better visualization"""
    texture = load_img(path)
    texture = np.tile(texture, (n, m, 1))

    return texture
