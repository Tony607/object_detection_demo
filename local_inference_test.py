#!/usr/bin/env python
# coding: utf-8

import os
import glob
import time
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image
from object_detection.utils import ops as utils_ops


if __name__ == "__main__":
    import argparse

    # python argparse_test.py 5 -v --color RED
    parser = argparse.ArgumentParser(
        description="TensorFlow Inference speed benchmark for object detection model."
    )
    # parser.add_argument("-v", "--verbose", help="increase output verbosity",
    #                     action="store_true")
    parser.add_argument(
        "--model",
        help="Path to the frozen graph .pb file.",
        type=str,
        default="./models/frozen_inference_graph.pb",
    )

    parser.add_argument(
        "--cpu", help="Force to use CPU during inference.", action="store_true"
    )
    parser.add_argument("--img", help="Path to a sample image to inference.", type=str)
    args = parser.parse_args()

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = args.model

    image_path = args.img

    assert os.path.isfile(PATH_TO_CKPT)
    assert os.path.isfile(image_path)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name="")

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return (
            np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        )

    def run_inference_benchmark(image, graph, trial=20, gpu=True):
        """Run TensorFlow inference benchmark.
        
        Arguments:
            image {np.array} -- Input image as an Numpy array.
            graph {tf.Graph} -- TensorFlow graph object.
        
        Keyword Arguments:
            trial {int} -- Number of inference to run for averaging. (default: {20})
            gpu {bool} -- Use Nvidia GPU when available. (default: {True})
        
        Returns:
            int -- Frame per seconds benchmark result.
        """

        with graph.as_default():
            if gpu:
                config = tf.ConfigProto()
            else:
                config = tf.ConfigProto(device_count={"GPU": 0})
            with tf.Session(config=config) as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    "num_detections",
                    "detection_boxes",
                    "detection_scores",
                    "detection_classes",
                    "detection_masks",
                ]:
                    tensor_name = key + ":0"
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name
                        )
                if "detection_masks" in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict["detection_boxes"], [0])
                    detection_masks = tf.squeeze(tensor_dict["detection_masks"], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(
                        tensor_dict["num_detections"][0], tf.int32
                    )
                    detection_boxes = tf.slice(
                        detection_boxes, [0, 0], [real_num_detection, -1]
                    )
                    detection_masks = tf.slice(
                        detection_masks, [0, 0, 0], [real_num_detection, -1, -1]
                    )
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1]
                    )
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8
                    )
                    # Follow the convention by adding back the batch dimension
                    tensor_dict["detection_masks"] = tf.expand_dims(
                        detection_masks_reframed, 0
                    )
                image_tensor = tf.get_default_graph().get_tensor_by_name(
                    "image_tensor:0"
                )

                # Run inference
                times = []
                # Kick start the first inference which takes longer and followings.
                output_dict = sess.run(
                    tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)}
                )
                for i in range(trial):
                    start_time = time.time()
                    output_dict = sess.run(
                        tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)}
                    )
                    delta = time.time() - start_time
                    times.append(delta)
                mean_delta = np.array(times).mean()
                fps = 1 / mean_delta
                print("average(sec):{:.3f},fps:{:.2f}".format(mean_delta, fps))

        return fps

    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection benchmark.
    fps = run_inference_benchmark(image_np, detection_graph, trial=20, gpu=not args.cpu)
