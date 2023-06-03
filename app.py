from flask import (
    Flask,
    render_template,
    request,
    send_file,
    redirect,
    jsonify,
    make_response,
)
from PIL import Image
import os
import cv2
import numpy as np
import json
from room_processing import *
from texture_mapping import get_wall_corners, map_texture, load_texture, image_resize
from wall_segmentation.segmenation import wall_segmenting, build_model
from wall_estimation.estimation import wall_estimation

app = Flask(__name__)

IMG_FOLDER = os.path.join("static", "IMG")
DATA_FOLDER = os.path.join("static", "data")

ROOM_IMAGE = os.path.join(IMG_FOLDER, "room.jpg")
COLORED_ROOM_PATH = os.path.join(IMG_FOLDER, "colored_room.jpg")
TEXTURED_ROOM_PATH = os.path.join(IMG_FOLDER, "textured_room.jpg")
TEXTURE_PATH = os.path.join(IMG_FOLDER, "texture.jpg")
MASK_PATH = os.path.join(DATA_FOLDER, "image_mask.npy")
CORNERS_PATH = os.path.join(DATA_FOLDER, "corners_estimation.npy")

# Load pretrained wall segmentation model
model = build_model()


# Home route
@app.route("/")
def main():
    return render_template("index.html", room=ROOM_IMAGE)


# Run two segmentation models for wall segmentation and wall position estimation
@app.route("/prediction", methods=["POST"])
def predict_image_room():
    try:
        if request.method == "POST":
            if "file" in request.files and request.files["file"].filename != "":
                img_path = request.files["file"].stream

                op_img = Image.open(img_path)
                op_img = np.asarray(op_img)

                if op_img.shape[0] > 600:
                    op_img = image_resize(op_img, height=600)

                op_img = Image.fromarray(op_img)

                op_img.save(ROOM_IMAGE)
                op_img.save(COLORED_ROOM_PATH)
                op_img.save(TEXTURED_ROOM_PATH)

                path_image = os.path.abspath(ROOM_IMAGE)

                # start wall segmention
                mask1 = wall_segmenting(model, path_image)

                # start wall estimation
                estimation_map = wall_estimation(path_image)
                # get coordinates of walls
                corners = get_wall_corners(estimation_map)

                # intersect two segmentation masks for getting somewhat more uniform result
                mask2 = np.full(mask1.shape, 0, dtype=np.uint8)

                for pts in corners:
                    pts = np.array(pts)
                    cv2.fillPoly(mask2, [pts], color=(255, 0, 0))

                mask2 = np.bool_(mask2)

                mask = mask1 & mask2

                with open(MASK_PATH, "wb") as f:
                    np.save(f, mask)

                with open(CORNERS_PATH, "wb") as f:
                    np.save(f, np.array(corners))

            if request.form["button"] == "color":
                return redirect("/colored_room")

            elif request.form["button"] == "texture":
                return redirect("/textured_room")

            return render_template("result.html")

    except Exception as e:
        error = "Error"
        return render_template("result.html", err=error)


@app.route("/colored_room", methods=["GET", "POST"])
def apply_color():
    img_path = ROOM_IMAGE

    img = load_img(img_path)

    save_image(img, COLORED_ROOM_PATH)

    return render_template("applied_color.html", new_room=COLORED_ROOM_PATH)


@app.route("/result_colored", methods=["GET", "POST"])
def result_colored():
    # load and process image
    img = load_img(ROOM_IMAGE)

    data = json.loads(request.data)
    if "color" in data and os.path.isfile(MASK_PATH):
        color = data.get("color")
        color = hex_to_rgb(color)

        # load segmentation mask of room walls
        with open(MASK_PATH, "rb") as f:
            mask = np.load(f)

        wall_color = img.copy()

        # change each wall's pixel on selected rgb value
        wall_color[np.where(mask != 0)] = color

        # transfer shadows and shine from original image
        img = brightness_transfer(img, wall_color, mask)

    # save processed image
    save_image(img, COLORED_ROOM_PATH)

    return make_response(jsonify(state="success", room_path=COLORED_ROOM_PATH), 200)


@app.route("/textured_room", methods=["GET", "POST"])
def textured_room():
    return render_template("applied_texture.html", new_room=TEXTURED_ROOM_PATH)


@app.route("/result_textured", methods=["GET", "POST"])
def result_textured():
    # save uploaded texture
    if "file" in request.files and request.files["file"].filename != "":
        img_path = request.files["file"].stream

        op_img = Image.open(img_path)
        op_img.save(TEXTURE_PATH)

    # load and process image
    img = load_img(ROOM_IMAGE)

    # load computed vertices of a wall
    with open(CORNERS_PATH, "rb") as f:
        corners = np.load(f)

    # load segmentation mask of room walls
    with open(MASK_PATH, "rb") as f:
        mask = np.load(f)

    # extend texture for better visualization
    texture = load_texture(TEXTURE_PATH, 6, 6)

    # perspective projection texture on walls
    img_textured = map_texture(texture, img, corners, mask)

    # transfer shadows and shine from original image
    out = brightness_transfer(img, img_textured, mask)

    # save processed image
    save_image(out, TEXTURED_ROOM_PATH)

    return make_response(jsonify(state="success", room_path=TEXTURED_ROOM_PATH), 200)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    rgb_tuple = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return rgb_tuple


if __name__ == "__main__":
    app.run(port=9000, debug=True)
