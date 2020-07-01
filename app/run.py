# import libraries
import base64
from io import BytesIO
import io
import time
import json
from imageio import imread, imwrite
from scipy import ndimage, misc
import data
import math

from fastai import *
from fastai.basic_train import load_learner
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Sampler, BatchSampler

from fastai.vision import open_image
from fastai.vision import *

# import Flask
from flask import Flask
from flask import render_template, request

import sys

sys.path.insert(0, "../")

from utils import *


class JSONImageItemList(ImageList):
    def open(self, fn):
        with io.open(fn) as f:
            j = json.load(f)
        drawing = list2drawing(j["drawing"], size=128)
        tensor = drawing2tensor(drawing)
        return Image(tensor.div_(255))


app = Flask(__name__)

# load model
model = load_learner("../")

class_labels = data.class_return()

# index webpage receives user input for the model
@app.route("/")
@app.route("/index")
def index():
    # render web page
    return render_template("index.html")


@app.route("/go/<dataURL>")
def pred(dataURL):
    """
    Render prediction result.
    """

    # decode base64  '._-' -> '+/='
    dataURL = dataURL.replace(".", "+")
    dataURL = dataURL.replace("_", "/")
    dataURL = dataURL.replace("-", "=")

    # get the base64 string
    image_b64_str = dataURL
    # convert string to bytes

    byte_data = base64.b64decode(image_b64_str)
    image_data = BytesIO(byte_data)
    img = imread(image_data)
    imwrite('image.jpg', img[:,:,:3])   
    image = cv2.imread('./image.jpg',cv2.IMREAD_COLOR)
    size = 256
    img = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    rgb = rgb.transpose(2,0,1).astype(np.float32)
    tensor = torch.from_numpy(rgb)
    image = Image(tensor.div_(255))

    # # apply model and print prediction
    _, _, preds = model.predict(image)

    # render the hook.html passing prediction resuls
    return render_template(
        "hook.html",
        result=class_labels[np.argmax(preds)],  # predicted class label
        dataURL=dataURL,  # image to display with result
        accuracy = math.ceil(torch.max(preds).item() * 100) 
    )


def main():
    app.run(host="localhost", port=3001, debug=True)


if __name__ == "__main__":
    main()
