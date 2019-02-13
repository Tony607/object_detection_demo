# [How to train an object detection model easy for free](https://www.dlology.com/blog/how-to-train-an-object-detection-model-easy-for-free/) | DLology Blog


## How to Run

Easy way: run [this Colab Notebook](https://colab.research.google.com/drive/1th1ub4iM-8wXZxAZzjBBN0rbpjzT8H9p).

Alternatively, if you want to use your images instead of ones comes with this repo.

Require [Python 3.5+](https://www.python.org/ftp/python/3.6.4/python-3.6.4.exe) installed.
### Fork and clone this repository to your local machine.
```
https://github.com/Tony607/object_detection_demo
```
### Install required libraries
`pip3 install -r requirements.txt`


### Step 1: Annotate some images
- Save some photos with your custom object(s), ideally with `jpg` extension to `./data/raw` directory. (If your objects are simple like ones come with this repo, 20 images can be enough.)
- Resize those photo to uniformed size. e.g. `(800, 600)` with
```
python resize_images.py --raw-dir ./data/raw --save-dir ./data/images --ext jpg --target-size "(800, 600)"
```
Resized images locate in `./data/images/`
- Train/test split those files into two directories, `./data/images/train` and `./data/images/test`

- Annotate reized images with [labelImg](https://tzutalin.github.io/labelImg/), generate `xml` files inside `./data/images/train` and `./data/images/test` folders. 

*Tips: use shortcuts (`w`: draw box, `d`: next file, `a`: previous file, etc.) to accelerate the annotation.*

- Commit and push your annotated images and xml files (`./data/images/train` and `./data/images/test`) to your forked repository.


### Step 2: Open [Colab notebook](https://colab.research.google.com/drive/1th1ub4iM-8wXZxAZzjBBN0rbpjzT8H9p)
- Replace the repository's url to yours and run it.


## How to run inference on frozen TensorFlow graph

Requirements:
- `frozen_inference_graph.pb` Frozen TensorFlow object detection model downloaded from Colab after training.
- `label_map.pbtxt` File used to map correct name for predicted class index downloaded from Colab after training.

Run the following Jupyter notebook locally.
```
local_inference_test.ipynb
```

## How to deploy the trained custom object detection model with OpenVINO

Requirements:
- Frozen TensorFlow object detection model. i.e. `frozen_inference_graph.pb` downloaded from Colab after training.
- The modified pipline config file used for training. Also downloaded from Colab after training.

Run the following Jupyter notebook locally and follow the instructions in side.
```
deploy/openvino_convert_tf_object_detection.ipynb
```